import torch

from resnet import *
from vgg import MyVGG
from loss import LossCalculator
from evaluate import accuracy, test_step
from utils import AverageMeter, get_data_set
from optimizer import get_optimizer
from datetime import datetime


def train_network(network, args):
    if network is None:
        if args.network == 'vgg':
            network = MyVGG()
        elif args.network == 'resnet':
            network = resnet32()

    device = torch.device("cuda" if args.gpu_flag is True else "cpu")
    network = network.to(device)
    optimizer, scheduler = get_optimizer(network, args)

    train_data_set = get_data_set(args, train_flag=True)
    test_data_set = get_data_set(args, train_flag=False)
    train_data_loader = torch.utils.data.DataLoader(train_data_set, batch_size=args.batch_size, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(test_data_set, batch_size=args.batch_size, shuffle=False)

    print("-*-" * 10 + "\n\t\tTrain network\n" + "-*-" * 10)
    for epoch in range(0, args.epoch):
        train_step(network, train_data_loader, test_data_loader, optimizer, device, epoch)
        if scheduler is not None:
            scheduler.step()

    return network


def train_step(network, train_data_loader, test_data_loader, optimizer, device, epoch):
    network.train()
    # set benchmark flag to faster runtime
    torch.backends.cudnn.benchmark = True

    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_calculator = LossCalculator()

    prev_time = datetime.now()

    for iteration, (inputs, targets) in enumerate(train_data_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = network(inputs)
        loss = loss_calculator.calc_loss(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

    cur_time = datetime.now()
    h, remainder = divmod((cur_time - prev_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    time_str = "Time: {:0>2d}:{:0>2d}:{:0>2d}".format(h, m, s)

    train_acc_str = '[Train] Top1: %2.4f, Top5: %2.4f, ' % (top1.avg, top5.avg)
    train_loss_str = 'Loss: %.4f. ' % loss_calculator.get_loss_log()

    test_top1, test_top5, test_loss = test_step(network, test_data_loader, device)

    test_acc_str = '[Test] Top1: %2.4f, Top5: %2.4f, ' % (test_top1, test_top5)
    test_loss_str = 'Loss: %.4f. ' % test_loss

    print('Epoch %d. ' % epoch + train_acc_str + train_loss_str + test_acc_str + test_loss_str + time_str)

    return None
