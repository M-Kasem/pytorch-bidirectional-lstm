# taken from here: https://github.com/pytorch/examples/blob/master/mnist/main.py
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import numpy as np
import math
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        # self.fc1 = nn.Linear(320, 50)
        # self.fc2 = nn.Linear(50, 10)
        

        self.nhid = 128
        self.nlayers = 28
        self.input_size = 28
        self.n_classes = 10

        # self.encoder = nn.Embedding(self.input_size, self.input_size)

        self.lstm = nn.LSTM(self.input_size, self.nhid, self.nlayers)
        self.decoder = nn.Linear(self.nhid, self.n_classes)

        self.init_weights()


    def forward(self, input, hidden):
    # def forward(self, x):
    #     x = F.relu(F.max_pool2d(self.conv1(x), 2))
    #     x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
    #     x = x.view(-1, 320)
    #     x = F.relu(self.fc1(x))
    #     x = F.dropout(x, training=self.training)
    #     x = self.fc2(x)
    #     return F.log_softmax(x)
        # print(input.size())
        output, hidden = self.lstm(input, hidden)
        # print(output.size())
        # print(hidden[0].size())
        # print(hidden[1].size())
        decoded = self.decoder( output[-1, :, :] )
        # print(decoded.size())
        return F.log_softmax(decoded), hidden
        # exit()


    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))

    def init_weights(self):
        initrange = 0.5
        # initrange = 1.0 / math.sqrt(self.nhid)
        self.lstm.named_parameters()
        for name, val in self.lstm.named_parameters():
            if name.find('bias') == -1:
                getattr(self.lstm, name).data.uniform_(-initrange, initrange)
            else:
                getattr(self.lstm, name).data.fill_(0)
            
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

def train(epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        hidden = model.init_hidden(args.batch_size)
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        print(data.size())
        # print(target.size())
        # print(target)
        # exit()

        # optimizer.zero_grad()

        # data = data.view(28, args.batch_size, 28) 
        data = data.permute(2,0,3,1).contiguous().view(28,args.batch_size,28)
        hidden = repackage_hidden(hidden)
        output, hidden = model(data, hidden)
        # output = model(data)

        loss = F.nll_loss(output, target)
        loss.backward()

        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        for par in model.parameters():
            par.data.add_(-args.lr, par.grad.data)
        # optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * args.batch_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
            print('Target: ', target)
            print('Train data: ', data[10, :, 10])

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        hidden = model.init_hidden(args.test_batch_size)
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)


        # output = model(data)
        # data = data.view(28, args.batch_size, 28)
        data = data.permute(2,0,3,1).contiguous().view(28,args.batch_size,28)

        output, hidden = model(data, hidden)

        # print(output.size())
        # for i in model.parameters():
            # print(i.size())
        # for name, param in model.named_parameters():
        #     print(name, param.size())
        # # print( tmp.shape )


        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        print('nll loss: ', F.nll_loss(output, target, size_average=False).data, ' .data[0]: ', F.nll_loss(output, target, size_average=False).data[0])
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        # while True:
        #     command = raw_input('Enter variable: ')
        #     if command == '':
        #         break
        #     try:
        #         # eval('print('+command+')')
        #         eval(command)
        #     except Exception as e:
        #         print(e)
        #         pass

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--no-shuffle', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('--exp_index', default=0, type=int, metavar='N',
                    help='gpu index')
    parser.add_argument('--job_id', type=int, metavar='N',
                    help='slurm job id for checkpoints identification')
    args = parser.parse_args()
    args.test_batch_size = args.batch_size
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)


    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle = not args.no_shuffle, drop_last=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=False, drop_last=True, **kwargs)
    model = Net()
    if args.cuda:
        model.cuda()

    train_loader.dataset.train_data = train_loader.dataset.train_data[:4*1, :, :]
    train_loader.dataset.train_labels = train_loader.dataset.train_labels[:4*1]
    
    test_loader.dataset.test_data = train_loader.dataset.train_data[:4*1, :, :]
    test_loader.dataset.test_labels = train_loader.dataset.train_labels[:4*1]
    # exit()

    print("Len train loader: ", len(train_loader), " Len train loader.data: ", len(train_loader.dataset))
    print("Len test loader: ", len(test_loader), " Len test loader.data: ", len(test_loader.dataset))
    print("train batch size: ", args.batch_size, " test batch size: ", args.batch_size)
    print("learning rate: ", args.lr, " shuffle train ", not args.no_shuffle)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test()