from __future__ import print_function
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.utils.data.distributed
import horovod.torch as hvd

import collections
import random, time, os

import torch

from utils import read_words, create_batches, to_var
from gated_cnn import GatedCNN

import torch.nn.functional as F
from torch.utils.data import DistributedSampler, DataLoader
from torch.nn.parallel import DistributedDataParallelCPU, DistributedDataParallel, DataParallel
import torch.multiprocessing as mp
import torch.distributed as dist
from model_2 import SomeNet

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=80, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 1)')
parser.add_argument('--lr', type=float, default=3e-3, metavar='LR',
                    help='learning rate (default: 3e-3)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


vocab_size      = 2000
seq_len         = 21
embd_size       = 200
n_layers        = 10
kernel          = (5, embd_size)
out_chs         = 64
res_block_count = 5
batch_size      = 80
rank            = 0
world_size      = 2

# Horovod: initialize library.
hvd.init()
torch.manual_seed(args.seed)

if args.cuda:
    # Horovod: pin GPU to local rank.
    torch.cuda.set_device(hvd.local_rank())
    torch.cuda.manual_seed(args.seed)


# Horovod: limit # of CPU threads to be used per worker.
torch.set_num_threads(1)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

words = read_words('/users/PAS1588/liuluyu0378/lab1/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled', seq_len, kernel[0])
word_counter = collections.Counter(words).most_common(vocab_size-1)
vocab = [w for w, _ in word_counter]
w2i = dict((w, i) for i, w in enumerate(vocab, 1))
w2i['<unk>'] = 0
print('vocab_size', vocab_size)
print('w2i size', len(w2i))

data = [w2i[w] if w in w2i else 0 for w in words]
data = create_batches(data, batch_size, seq_len)
split_idx = int(len(data) * 0.8)
training_data = data[:split_idx]
test_data = data[split_idx:]

rank = hvd.rank()
training_length = len(training_data)
test_length = len(test_data)
training_data = training_data[int(rank * training_length / hvd.size()): int((rank + 1)* training_length / hvd.size())]
test_data = test_data[int(rank * test_length / hvd.size()): int((rank + 1)* test_length / hvd.size())]


print('train samples:', len(training_data))
print('test samples:', len(test_data))

train_dataset = training_data
# Horovod: use DistributedSampler to partition the training data.
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs)

test_dataset = test_data
# Horovod: use DistributedSampler to partition the test data.
test_sampler = torch.utils.data.distributed.DistributedSampler(
    test_dataset, num_replicas=hvd.size(), rank=hvd.rank())
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                                          sampler=test_sampler, **kwargs)


model = SomeNet(seq_len, vocab_size, embd_size, n_layers, kernel, out_chs, res_block_count, vocab_size)

if args.cuda:
    # Move model to GPU.
    model.cuda()

# Horovod: scale learning rate by the number of GPUs.
optimizer = optim.Adam(model.parameters(), lr=args.lr * hvd.size())

# Horovod: broadcast parameters & optimizer state.
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

# Horovod: (optional) compression algorithm.
compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

# Horovod: wrap optimizer with DistributedOptimizer.
optimizer = hvd.DistributedOptimizer(optimizer,
                                     named_parameters=model.named_parameters(),
                                     compression=compression)


def train(epoch):
    model.train()
    # Horovod: set epoch to sampler for shuffling.
    train_sampler.set_epoch(epoch)
    for batch_idx, (data, target) in enumerate(train_dataset):
        a = time.time()
        # for i in range(len(data)):
        #     print(len(data[0][0]))
        #     data[i] = to_var(torch.stack(data[i]))
            
            
        # data = torch.stack(data)
        # target = torch.stack(target)
        
        data = to_var(torch.LongTensor(data)) # (bs, seq_len)
        target = to_var(torch.LongTensor(target)) # (bs,)
        # print( data.size(), target.size())
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = F.cross_entropy(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        b = time.time()
        if batch_idx % args.log_interval == 0:
            # Horovod: use train_sampler to determine the number of examples in
            # this worker's partition.
            if hvd.rank() == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_dataset),
                    100. * batch_idx / len(train_dataset), loss.item()))
                print("Train time: ", b -a)
    


def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()


def test():
    model.eval()
    test_loss = 0.
    test_accuracy = 0.
    counter = 0
    correct = 0
    for data, target in test_dataset:
        
        data = to_var(torch.LongTensor(data)) # (bs, seq_len)
        target = to_var(torch.LongTensor(target)) # (bs,)
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target).item()
        _, pred_ids = torch.max(output, 1)
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += torch.sum(pred_ids == target).data.item()
        
        counter += data.size(0)
        test_accuracy += pred.eq(target.data.view_as(pred)).cpu().float().sum()
        
    # print('Test Acc: {:.2f} % ({}/{})'.format(100 * correct / counter, correct, counter))
    # print('Test Loss: {:.4f}'.format(losses/counter))

    # # Horovod: use test_sampler to determine the number of examples in
    # # this worker's partition.
    test_loss /= counter
    test_accuracy /= counter

    # Horovod: average metric values across workers.
    test_loss = metric_average(test_loss, 'avg_loss')
    test_accuracy = metric_average(test_accuracy, 'avg_accuracy')

    # Horovod: print output only on first rank.
    if hvd.rank() == 0:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
            test_loss, 100. * test_accuracy))


for epoch in range(1, args.epochs + 1):
    aa = time.time()
    train(epoch)
    bb = time.time()
    print("************* Total train time: ", bb - aa, "***************")
    test()
