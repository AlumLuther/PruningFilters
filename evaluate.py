import torch

from loss import LossCalculator
from utils import AverageMeter, get_data_set


def test_network(network, args):
    if network is None:
        return

    device = torch.device("cuda" if args.gpu_flag is True else "cpu")
    network.to(device)

    data_set = get_data_set(args, train_flag=False)
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=100, shuffle=False)

    test_top1, test_top5, test_loss = test_step(network, data_loader, device)

    print("-*-" * 10 + "\n\t\tTest network\n" + "-*-" * 10)
    test_acc_str = 'Top1: %2.4f, Top5: %2.4f, ' % (test_top1, test_top5)
    test_loss_str = 'Loss: %.4f. ' % test_loss
    print(test_acc_str + test_loss_str)

    return


def test_step(network, data_loader, device):
    network.eval()

    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_calculator = LossCalculator()

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = network(inputs)
            loss_calculator.calc_loss(outputs, targets)
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

    return top1.avg, top5.avg, loss_calculator.get_loss_log()


def accuracy(output, target, topk=(1,)):
    """
        Computes the precision@k for the specified values of k
        ref: https://github.com/chengyangfu/pytorch-vgg-cifar10
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
