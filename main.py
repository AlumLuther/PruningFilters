import torch

from network import MyVGG, save_network
from parameter import get_parameter
from train import train_network
from evaluate import test_network
from prune import prune_network

if __name__ == '__main__':
    network = None
    args = get_parameter()
    if args.load_path:
        check_point = torch.load(args.load_path)
        network = MyVGG(check_point['cfg'])
        network.load_state_dict(check_point['state_dict'])

    if args.train_flag:
        network = train_network(network, args)
    elif args.prune_flag:
        network = prune_network(network, args)
    elif args.test_flag:
        network = test_network(network, args)

    save_network(network, args.save_path)
