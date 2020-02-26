from parameter import get_parameter
from utils import load_network, save_network
from train import train_network
from evaluate import test_network
from hardprune import hard_prune_network
from softprune import soft_prune_network

if __name__ == '__main__':
    args = get_parameter()

    network = load_network(args)

    if args.train_flag:
        network = train_network(network, args)
    elif args.hard_prune_flag:
        network = hard_prune_network(network, args)
    elif args.soft_prune_flag:
        network = soft_prune_network(network, args)

    print(network)
    test_network(network, args)
    network = train_network(network, args)
    save_network(network, args)
