import os
from datetime import datetime
import argparse
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist
import collections
import random, time, os

from utils import read_words, create_batches, to_var
from gated_cnn import GatedCNN

import torch.nn.functional as F
from torch.utils.data import DistributedSampler, DataLoader
from torch.nn.parallel import DistributedDataParallelCPU, DistributedDataParallel, DataParallel
import torch.multiprocessing as mp
import torch.distributed as dist

vocab_size      = 2000
seq_len         = 21
embd_size       = 200
n_layers        = 10
kernel          = (5, embd_size)
out_chs         = 64
res_block_count = 5
batch_size      = 80
rank            = 0
# world_size      = torch.cuda.device_count()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default=2, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '8888'
    print("starting")
    mp.spawn(train, nprocs=args.gpus, args=(args,))
    print("finished")


def train(gpu, args):
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    torch.manual_seed(0)

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
    
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)
    # Wrap the model
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    print("model transfered")
    
    optimizer = torch.optim.Adadelta(model.parameters())
    loss_fn = nn.NLLLoss()
    # Data loading code

                                               
    train_sampler = torch.utils.data.distributed.DistributedSampler(training_data,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)
    train_loader = torch.utils.data.DataLoader(dataset=training_data,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True,
                                               sampler=train_sampler)

    start = datetime.now()
    total_step = len(train_loader)

    print("loaded")
    for epoch in range(args.epochs):
        a = time.time()
        print('----epoch', epoch)
        # random.shuffle(data)
        # print(len(data))
        for batch_ct, (X, Y) in enumerate(train_loader):
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
        print('current performance at epoch', epoch, "time:", b-a)

    if gpu == 0:
        print("Training complete in: " + str(datetime.now() - start))


if __name__ == '__main__':
    main()