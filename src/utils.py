import copy
import torch
import numpy as np
import random
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset
from torchvision.datasets import ImageFolder
import math

def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum(np.exp(x))
    return ex/sum_ex

def merge_user_data(user_data_list):
    merged_data = []
    user_data_indices = []

    for user_index, user_data in enumerate(user_data_list):
        merged_data.extend(user_data)
        user_indices = [user_index] * len(user_data)
        user_data_indices.extend(user_indices)

    return merged_data, user_data_indices

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_dataset(args):
    if args.dataset == 'synthetic':
        dimension = 60
        NUM_CLASS = 10
        NUM_USER = args.num_users
        alpha = 0.5
        beta = 0.5
        iid = args.iid
        setup_seed(args.seed)
        samples_per_user = np.random.randint(5000, 10000, size=(NUM_USER,))
        num_samples = np.sum(samples_per_user)

        X_split = []
        y_split = []
        user_groups = [[] for _ in range(NUM_USER)]


        mean_W = np.random.normal(0, alpha, NUM_USER)
        mean_b = mean_W
        B = np.random.normal(0, beta, NUM_USER)
        mean_x = np.zeros((NUM_USER, dimension))

        diagonal = np.zeros(dimension)
        for j in range(dimension):
            diagonal[j] = np.power((j + 1), -1.2)
        cov_x = np.diag(diagonal)

        for i in range(NUM_USER):
            if iid == 1:
                mean_x[i] = np.ones(dimension) * B[i]
            else:
                mean_x[i] = np.random.normal(B[i], 1, dimension)

        if iid == 1:
            W_global = np.random.normal(0, 1, (dimension, NUM_CLASS))
            b_global = np.random.normal(0, 1, NUM_CLASS)

        pointer = 0
        for i in range(0, NUM_USER):

            W = np.random.normal(mean_W[i], 1, (dimension, NUM_CLASS))
            b = np.random.normal(mean_b[i], 1, NUM_CLASS)

            if iid == 1:
                W = W_global
                b = b_global

            xx = np.random.multivariate_normal(mean_x[i], cov_x, samples_per_user[i])
            yy = np.zeros(samples_per_user[i])

            for j in range(samples_per_user[i]):
                tmp = np.dot(xx[j], W) + b
                yy[j] = np.argmax(softmax(tmp))

            xx = xx.tolist()
            yy = yy.tolist()

            X_split.extend(xx)
            y_split.append(yy)
            user_groups[i] = list(range(pointer, pointer + samples_per_user[i]))
            pointer = pointer + samples_per_user[i]
        label_list = [item for sublist in y_split for item in sublist]
        features_tensor = torch.tensor(X_split, dtype=torch.float32)
        labels_tensor = torch.tensor(label_list, dtype=torch.int64)
        combined_data = TensorDataset(features_tensor, labels_tensor)

    elif args.dataset == 'mnist' or args.dataset == 'fmnist' or args.dataset == 'mmnist':
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'

            apply_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])

            train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

            test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)
            combined_data = ConcatDataset([train_dataset, test_dataset])
        elif args.dataset == 'mmnist':
            data_dir = '../data/mmnist/'
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.3), (0.3))
            ])

            combined_data = ImageFolder(data_dir, transform=transform)
        else:
            data_dir = '../data/fmnist/'
            apply_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])

            train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

            test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)
            combined_data = ConcatDataset([train_dataset, test_dataset])

        if args.iid:
            user_groups = mnist_iid(combined_data, args.num_users)
        else:
            user_groups = mnist_noniid(args.dataset, combined_data, args.num_users, args.q)

    return combined_data, user_groups

def average_loss_acc(local_model, num_users):
    train_loss_personal_local, train_loss_global_local = 0, 0
    test_acc_personal_local, test_acc_global_local = 0, 0
    for idx in range(num_users):
        train_loss_personal_local += local_model[idx].train_personal_loss
        train_loss_global_local += local_model[idx].train_global_loss
        test_acc_personal_local += local_model[idx].test_acc_personal
        test_acc_global_local += local_model[idx].test_acc_global

    train_loss_personal_avg = train_loss_personal_local/num_users
    train_loss_global_avg = train_loss_global_local/num_users
    test_acc_personal_avg = test_acc_personal_local/num_users
    test_acc_global_avg = test_acc_global_local/num_users
    return train_loss_global_avg, train_loss_personal_avg, test_acc_personal_avg, test_acc_global_avg

def average_loss_acc_centralized(local_model, num_users):
    train_loss_local, test_acc_local = 0, 0
    for idx in range(num_users):
        train_loss_local += local_model[idx].train_loss
        test_acc_local += local_model[idx].test_acc

    train_loss_avg = train_loss_local/num_users
    test_acc_avg = test_acc_local/num_users
    return train_loss_avg, test_acc_avg


def exp_details(args):
    print('\nParameter description')
    print(f'    Model              : {args.model}')
    print(f'    Optimizer          : {args.optimizer}')
    print(f'    Framework          : {args.framework}')
    print(f'    Client selection   : {args.strategy}\n')
    print(f'    Global Rounds      : {args.epochs}\n')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    dataset            : {args.dataset}')
    print(f'    Num of users       : {args.num_users}')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Learning  Rate     : {args.lr}')
    print(f'    Lambda:            : {args.Lambda}')
    print(f'    rho                : {args.rho}')
    print(f'    mu                 : {args.mu}')
    print(f'    Local Epochs       : {args.local_ep}')
    print(f'    Local Batch size   : {args.local_bs}\n')
    return

def mnist_iid(dataset, num_users):
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset_name, dataset, num_users, q):
    if dataset_name == 'mmnist':
        labels = np.array(dataset.targets)
    else:
        labels_train = dataset.datasets[0].targets.numpy()
        labels_test = dataset.datasets[1].targets.numpy()
        labels = np.concatenate((labels_train,labels_test), axis=0)
    num_labels = len(np.unique(labels))

    num_shards = math.ceil(q * num_users / num_labels)

    indices = [np.where(labels == i)[0] for i in range(num_labels)]
    data_split_indices = []
    for i in range(num_labels):
        indices_i = np.array_split(indices[i], num_shards)
        data_split_indices.extend(indices_i)

    user_data_indices = {}
    for user_id in range(num_users):
        np.random.shuffle(data_split_indices)
        selected_indices = np.concatenate(data_split_indices[:q])
        user_data_indices[user_id] = selected_indices
        del data_split_indices[:q]

    return user_data_indices

