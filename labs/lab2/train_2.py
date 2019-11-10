import collections
import random, time, os

import torch
import torch.nn as nn

from utils import read_words, create_batches, to_var
from gated_cnn import GatedCNN

import torch.nn.functional as F
from torch.utils.data import DistributedSampler, DataLoader
from torch.nn.parallel import DistributedDataParallelCPU, DistributedDataParallel
import torch.multiprocessing as mp
import torch.distributed as dist
from utils.data_parallel import BalancedDataParallel


vocab_size      = 2000
seq_len         = 21
embd_size       = 200
n_layers        = 10
kernel          = (5, embd_size)
out_chs         = 64
res_block_count = 5
batch_size      = 80
rank            = 0
world_size      = torch.cuda.device_count()


def train(model, data, test_data, optimizer, loss_fn, n_epoch=5):
    print('=========training=========')
    setup(rank, world_size)
    n = torch.cuda.device_count() // world_size
    device_ids = list(range(rank * n, (rank + 1) * n))

    model.train()
    for epoch in range(n_epoch):
        a = time.time()
        print('----epoch', epoch)
        random.shuffle(data)
        print(len(data))
        for batch_ct, (X, Y) in enumerate(data):
            X = to_var(torch.LongTensor(X)) # (bs, seq_len)
            Y = to_var(torch.LongTensor(Y)) # (bs,)
            # print(X.size(), Y.size())
            # print(X)
            # print(batch_ct, X.size(), Y.size())
            pred = model(X) # (bs, ans_size)
            # _, pred_ids = torch.max(pred, 1)
            loss = loss_fn(pred, Y)
            if batch_ct % 100 == 0:
                print('loss: {:.4f}'.format(loss.data.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        b = time.time()
        print('current performance at ecpoh', epoch, "time:", b-a)
        test(model, test_data)


def test(model, data):
    model.eval()
    counter = 0
    correct = 0
    losses = 0.0
    for batch_ct, (X, Y) in enumerate(data):
        X = to_var(torch.LongTensor(X)) # (bs, seq_len)
        Y = to_var(torch.LongTensor(Y)) # (bs,)
        pred = model(X) # (bs, ans_size)
        loss = loss_fn(pred, Y)
        losses += torch.sum(loss).data.item()
        _, pred_ids = torch.max(pred, 1)
        # print('loss: {:.4f}'.format(loss.data[0]))
        correct += torch.sum(pred_ids == Y).data.item()
        counter += X.size(0)
    print('Test Acc: {:.2f} % ({}/{})'.format(100 * correct / counter, correct, counter))
    print('Test Loss: {:.4f}'.format(losses/counter))



if __name__ == "__main__":
    device = torch.device('cuda' if args.cuda else 'cpu')
    mp.set_start_method('spawn')
    distributed_mode = True
    
    # gpu_devices = ','.join([str(id) for id in world_size])
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
    # os.environ['MASTER_ADDR'] = '127.0.0.1'
    # os.environ['MASTER_PORT'] = '5446'
    # dist.init_process_group(backend='nccl',init_method='env://', world_size=world_size, rank=rank)
    
    # world_size (int, optional) – Number of processes participating in the job
    # init_method (str, optional) – URL specifying how to initialize the process group. Default is “env://” if no init_method or store is specified. Mutually exclusive with store.
    setup()

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
    print('train samples:', len(training_data))
    print('test samples:', len(test_data))

    model = GatedCNN(seq_len, vocab_size, embd_size, n_layers, kernel, out_chs, res_block_count, vocab_size)
    cuda = None
    if torch.cuda.is_available():
        print("cuda")
        model.cuda()
        cuda = True
    else:
        cuda = False

    if distributed_mode:
        # sampler = DistributedSampler(training_data, num_replicas=world_size, rank=rank)
        if cuda:
            model = DistributedDataParallel(model)
        else:
            model = DistributedDataParallelCPU(model)
    
        
    #non-distributed. set the model to DataParallel to increase training speed
    elif not distributed_mode and cuda:
        model = torch.nn.DataParallel(model)

optimizer = torch.optim.Adadelta(model.parameters())
loss_fn = nn.NLLLoss()
train(model, training_data, test_data, optimizer, loss_fn)
# test(model, test_data)
