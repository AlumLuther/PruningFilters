import os
import argparse


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu-flag', action='store_true',
                        help='flag for using gpu', default=True)

    parser.add_argument('--train-flag', action='store_true',
                        help='flag for training network', default=False)

    parser.add_argument('--prune-flag', action='store_true',
                        help='flag for pruning network', default=False)

    parser.add_argument('--test-flag', action='store_true',
                        help='flag for testing network', default=False)

    parser.add_argument('--network', type=str,
                        help='Network for training', default='vgg')

    parser.add_argument('--data-set', type=str,
                        help='Data set for training network', default='CIFAR10')

    parser.add_argument('--data-path', type=str,
                        help='Path of dataset', default='../')

    parser.add_argument('--epoch', type=int,
                        help='number of epoch for training network', default=10)

    parser.add_argument('--batch-size', type=int,
                        help='batch size', default=128)

    parser.add_argument('--lr', type=float,
                        help='learning rate', default=0.1)

    # ref: https://github.com/kuangliu/pytorch-cifar
    parser.add_argument('--lr-milestone', nargs='+', type=int,
                        help='list of epoch for adjust learning rate', default=None)

    parser.add_argument('--lr-gamma', type=float,
                        help='factor for decay learning rate', default=0.1)

    parser.add_argument('--momentum', type=float,
                        help='momentum for optimizer', default=0.9)

    parser.add_argument('--weight-decay', type=float,
                        help='factor for weight decay in optimizer', default=5e-4)

    parser.add_argument('--imsize', type=int,
                        help='size for image resize', default=None)

    parser.add_argument('--cropsize', type=int,
                        help='size for image crop', default=32)

    parser.add_argument('--crop-padding', type=int,
                        help='size for padding in image crop', default=4)

    parser.add_argument('--hflip', type=float,
                        help='probability of random horizontal flip', default=0.5)

    parser.add_argument('--load-path', type=str,
                        help='trained model load path to prune', default=None)

    parser.add_argument('--save-path', type=str,
                        help='model save path', required=True)

    parser.add_argument('--independent-prune-flag', action='store_true',
                        help='prune multiple layers by "independent strategy"', default=False)

    parser.add_argument('--prune-layers', nargs='+',
                        help='layer index for pruning', default=None)

    parser.add_argument('--prune-channels', nargs='+', type=int,
                        help='number of channel to prune layers', default=None)

    return parser


def get_parameter():
    parser = build_parser()
    args = parser.parse_args()

    print("-*-" * 10 + "\n\t\tArguments\n" + "-*-" * 10)
    for key, value in vars(args).items():
        print("%s: %s" % (key, value))

    save_folder = args.save_path[0:args.save_path.rindex('/')]
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print("Make dir: ", save_folder)

    return args
