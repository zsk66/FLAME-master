import os
import copy
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import PerLocalUpdate
from models import MLP, MLR, CNN, SVM
from utils import get_dataset, exp_details, setup_seed, average_loss_acc
import torchvision


if __name__ == '__main__':
    print("PyTorch Version:", torch.__version__)
    print("Torchvision Version:", torchvision.__version__)
    print("GPU is available?", torch.cuda.is_available())
    start_time = time.time()

    path_project = os.path.abspath('.')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    setup_seed(args.seed)
    print('random seed =', args.seed)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset, user_groups = get_dataset(args)

    local_model, model = [], []
    if args.model == 'MLP':
        global_model = MLP(args=args)
    elif args.model == 'MLR':
        global_model = MLR(args=args)
    elif args.model == 'SVM':
        global_model = SVM()
    elif args.model == 'CNN':
        global_model = CNN()
    else:
        exit('Error: unrecognized model')

    global_model.to(args.device)
    global_model.train()
    print(global_model)
    global_weights = global_model.state_dict()
    w = copy.deepcopy(global_weights)
    total_params = sum(p.numel() for p in global_model.parameters())
    print(f"Total number of parameters: {total_params}")
    train_loss_personal, test_acc_personal, train_loss_global, test_acc_global = [], [], [], []
    train_loss_personal_local, train_loss_global_local = 0, 0
    for idx in range(args.num_users):
        local_model.append(PerLocalUpdate(args=args, global_model=global_model, dataset=dataset, idxs=user_groups[idx], logger=logger))



    train_loss_global_avg, train_loss_personal_avg, test_acc_personal_avg, test_acc_global_avg = average_loss_acc(local_model, args.num_users)
    train_loss_personal.append(train_loss_personal_avg)
    train_loss_global.append(train_loss_global_avg)
    test_acc_personal.append(test_acc_personal_avg)
    test_acc_global.append(test_acc_global_avg)
    for epoch in tqdm(range(args.epochs)):

        gradients_L2, local_sum, local_train_losses_personal, local_test_accuracies_personal, \
        local_test_accuracies_global, local_train_losses_global = [], [], [], [], [], []

        m1 = max(int(args.frac_candidates * args.num_users), 1)
        m2 = max(int(args.frac * args.num_users), 1)

        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        idxs_candidates_users = np.random.choice(range(args.num_users), m1, replace=False)

        # Client selection:
        if args.strategy == 'biased':
            for idx in idxs_candidates_users:
                gradient_L2 = local_model[idx].calculate_gradient_l2_norm(w)
                gradients_L2.append((idx, gradient_L2))
                sorted_norms = sorted(gradients_L2, key=lambda x: x[1], reverse=True)
            idxs_users = [x[0] for x in sorted_norms[:m2]]
        elif args.strategy == 'random':
            idxs_users = np.random.choice(idxs_candidates_users, m2, replace=False)
        elif args.strategy == 'full':
            idxs_users = range(args.num_users)

        else:
            exit('Error: unrecognized client selection strategy.')

        print(f"\n \x1b[{35}m{'The IDs of selected clients:{}'.format(np.sort(idxs_users))}\x1b[0m")


        # Update local and global models
        for idx in idxs_users:
            lsum, local_test_acc_personal, local_train_loss_personal, local_test_acc_global\
                = local_model[idx].update_weights(global_round=epoch, global_model=global_model, w=copy.deepcopy(w), UserID=idx)

            local_test_accuracies_personal.append(copy.deepcopy(local_test_acc_personal))
            local_test_accuracies_global.append(copy.deepcopy(local_test_acc_global))
            local_sum.append(copy.deepcopy(lsum))


        #
        if args.framework == 'FLAME':
            for key in w.keys():
                w[key] = torch.zeros_like(w[key])
                for i in range(0, len(local_model)):
                    w[key] += (local_model[i].wi[key] + (1/args.rho)*local_model[i].alpha[key]) * 1/args.num_users
        elif args.framework == 'pFedMe':
            for key in w.keys():
                w[key] = torch.zeros_like(w[key])
                for i in range(0, len(local_model)):
                    w[key] += local_model[i].wi[key] * 1/args.num_users
        global_model.load_state_dict(w)

        train_loss_global_avg, train_loss_personal_avg, test_acc_personal_avg, test_acc_global_avg = average_loss_acc(
            local_model, args.num_users)
        train_loss_personal.append(train_loss_personal_avg)
        train_loss_global.append(train_loss_global_avg)
        test_acc_personal.append(test_acc_personal_avg)
        test_acc_global.append(test_acc_global_avg)

        print(f"\n\x1b[{34}m{'The average Test accuracy of personal model is {:.2f}%'.format(100 * test_acc_personal_avg)}\x1b[0m")
        print(f"\x1b[{34}m{'The average Test accuracy of global model is {:.2f}%'.format(100 * test_acc_global_avg)}\x1b[0m")
        print(f"\x1b[{34}m{'The average Training loss of personal model is {:.2f}'.format(train_loss_personal_avg)}\x1b[0m")
        print(f"\x1b[{34}m{'The average Training loss of global model is {:.2f}'.format(train_loss_global_avg)}\x1b[0m")


    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))





