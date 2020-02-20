import torch
from datetime import datetime

from loss import LossCalculator
from train import train_step
from vgg import MyVGG
from optimizer import get_optimizer
from hardprune import hard_prune_step
from utils import get_data_set, AverageMeter
from evaluate import accuracy, test_step


def soft_prune_network(network, args):
    if network is None:
        if args.network == 'vgg':
            network = MyVGG()

    layers = []
    channels = []
    num = 1

    for i in range(len(network.features)):
        if isinstance(network.features[i], torch.nn.Conv2d):
            layers.append('conv' + str(num))
            channels.append(int(round(network.features[i].out_channels * args.prune_rate)))
            num += 1

    network = soft_train(network, args)
    network = hard_prune_step(network, layers, channels, args.independent_prune_flag)

    print("-*-" * 10 + "\n\t\tPrune network\n" + "-*-" * 10)
    print(network)

    return network


def soft_train(network, args):
    device = torch.device("cuda" if args.gpu_flag is True else "cpu")
    optimizer, scheduler = get_optimizer(network, args)

    train_data_set = get_data_set(args, train_flag=True)
    test_data_set = get_data_set(args, train_flag=False)
    train_data_loader = torch.utils.data.DataLoader(train_data_set, batch_size=args.batch_size, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(test_data_set, batch_size=args.batch_size, shuffle=False)

    print("-*-" * 10 + "\n\t\tTrain network\n" + "-*-" * 10)
    for epoch in range(0, args.epoch):
        network = network.cpu()
        network = soft_prune_step(network, args.prune_rate)
        network = network.to(device)
        train_step(network, train_data_loader, test_data_loader, optimizer, device, epoch)
        if scheduler is not None:
            scheduler.step()

    return network


def soft_prune_step(network, prune_rate):
    for i in range(len(network.features)):
        if isinstance(network.features[i], torch.nn.Conv2d):
            kernel = network.features[i].weight.data
            sum_of_kernel = torch.sum(torch.abs(kernel.view(kernel.size(0), -1)), dim=1)
            _, args = torch.sort(sum_of_kernel)
            soft_prune_list = args[:int(round(kernel.size(0) * prune_rate))].tolist()
            for j in soft_prune_list:
                network.features[i].weight.data[j] = torch.zeros_like(network.features[i].weight.data[j])
                network.features[i].bias.data[j] = torch.zeros_like(network.features[i].bias.data[j])
    return network
