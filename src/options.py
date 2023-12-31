import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--framework', type=str, default='FLAME', choices=["FLAME", 'pFedMe'], help="type of federated learning framework")
    parser.add_argument('--num_users', type=int, default=100, help="number of users, must be a multiple of 5")
    parser.add_argument('--q', type=int, default=2, help="number of labels in each client")
    parser.add_argument('--model', type=str, default='SVM', choices=['MLP', 'MLR', 'CNN', 'SVM'], help='model name')
    parser.add_argument('--dataset', type=str, default='synthetic', choices=["mnist", "fmnist", 'mmnist', 'synthetic'], help="name of dataset")
    parser.add_argument('--strategy', type=str, default='full', choices=['biased', 'random', 'full'], help="client selection strategy")
    parser.add_argument('--frac_candidates', type=float, default=0.5, help='fraction of clients candidates: S')
    parser.add_argument('--frac', type=float, default=0.1, help='fraction of clients: C')
    parser.add_argument('--optimizer', type=str, default='sgd', help="type of optimizer")
    parser.add_argument('--momentum', type=float, default=0, help='SGD momentum (default: 0)')
    parser.add_argument('--epochs', type=int, default=100, help="total communication rounds")
    parser.add_argument('--local_ep', type=int, default=5, help="number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=50, help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--mu', type=float, default=0.01, help='hpy in regularization term')
    parser.add_argument('--Lambda', type=float, default=5, help='hpy in Moreau Envelope')
    parser.add_argument('--rho', type=float, default=0.01, help='hyp in Penalty term')
    parser.add_argument('--iid', type=int, default=0, help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--file_name', type=str, default='name', help='file name.')
    parser.add_argument('--seed', type=int, default=14, help="random seed")
    parser.add_argument('--eta', type=float, default=0.5, help="learning rate of global model in pFedMe")
    parser.add_argument('--verbose', type=int, default=0, help='verbose')
    args = parser.parse_args()
    return args
