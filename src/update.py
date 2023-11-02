import torch
import copy
import random
from torch import nn
from torch.utils.data import DataLoader, Dataset
from models import MLP, MLR, CNN, SVM
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):

        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class PerLocalUpdate(object):
    def __init__(self, args, global_model, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        if self.args.model == 'MLP':
            self.model = MLP(args=args)
        elif self.args.model == 'MLR':
            self.model = MLR(args=args)
        elif self.args.model == 'CNN':
            self.model = CNN()
        elif self.args.model == 'SVM':
            self.model = SVM()
        else:
            exit('Error: unrecognized model')
        self.model.to(self.args.device)
        self.model.train()
        self.wi = copy.deepcopy(global_model.state_dict())
        self.alpha = {}
        for key in self.wi.keys():
            self.alpha[key] = torch.zeros_like(self.wi[key]).to(self.args.device)
        self.trainloader, self.testloader = self.train_val_test(dataset, list(idxs))
        if self.args.model == 'SVM':
            self.criterion = nn.MultiMarginLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.train_personal_loss = self.inference(self.model)
        self.train_global_loss = self.inference(global_model)
        self.test_acc_personal = test_inference(self.args, model=self.model, testloader=self.testloader)
        self.test_acc_global = test_inference(self.args, model=global_model, testloader=self.testloader)
    def calculate_gradient_l2_norm(self, w):
        self.model.train()
        data_iterator = iter(self.trainloader)
        batch_data, batch_labels = next(data_iterator)
        if self.args.model == 'MLP' or self.args.model == 'MLR':
            batch_data = batch_data.reshape(-1, 784)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)
        self.model.zero_grad()
        output = self.model(batch_data)
        loss = self.criterion(output, batch_labels)
        loss.backward()
        total_norm = 0
        weights = copy.deepcopy(self.model.state_dict())

        for name, param in self.model.named_parameters():
            if param.requires_grad == True:
                param_norm = (param.grad + self.args.Lambda * (weights[name] - w[name])+self.args.mu*weights[name]).data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm

    def train_val_test(self, dataset, idxs):

        train_ratio = 0.8
        test_ratio = 0.2
        total_samples = len(idxs)
        train_size = int(train_ratio * total_samples)
        test_size = total_samples - train_size
        random.shuffle(idxs)
        idxs_train = idxs[:train_size]
        idxs_test = idxs[train_size+1:]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train), batch_size=self.args.local_bs, shuffle=True)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test), batch_size=self.args.local_bs, shuffle=True)
        return trainloader, testloader

    def update_weights(self, global_round, global_model, w, UserID):
        local_sum = {}
        self.model.train()
        hpy_lambda = self.args.Lambda/(self.args.num_users)

        if self.args.framework == 'pFedMe' or self.args.framework == 'FLAME':
            self.wi = copy.deepcopy(w)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr)
        epoch_loss = []
        alpha_prev = copy.deepcopy(self.alpha)
        wi_prev = copy.deepcopy(self.wi)


        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                if self.args.model == 'MLP' or self.args.model == 'MLR':
                    images = images.reshape(-1, 784)
                self.model.zero_grad()
                optimizer.zero_grad()
                log_probs = self.model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                theta_pre = copy.deepcopy(self.model.state_dict())
                if self.args.framework == 'FLAME' or self.args.framework == 'pFedMe':
                    for name, param in self.model.named_parameters():
                        if param.requires_grad == True:
                            param.grad = param.grad + \
                                         self.args.Lambda * (theta_pre[name]-self.wi[name]) + \
                                         self.args.mu * theta_pre[name]

                else:
                    continue
                optimizer.step()

                if self.args.verbose and (iter % 1 == 0) and (batch_idx % 10 == 0):
                    print('| Communication Round : {} | Client {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, UserID,
                        iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss))


        weights = self.model.state_dict()
        if self.args.framework == 'FLAME':
            for key in self.alpha.keys():
                self.wi[key] = 1/(hpy_lambda + self.args.rho) * \
                               (hpy_lambda * weights[key] + self.args.rho*w[key]-alpha_prev[key])
                self.alpha[key] = self.alpha[key] + self.args.rho * (self.wi[key] - w[key])


                local_sum[key] = (self.wi[key] - wi_prev[key]) + (1 / self.args.rho) * (
                            self.alpha[key] - alpha_prev[key])

        elif self.args.framework == 'pFedMe':
            for key in self.alpha.keys():
                self.wi[key] = self.wi[key] - self.args.eta * self.args.Lambda * (self.wi[key] - weights[key])
                local_sum[key] = (self.wi[key] - wi_prev[key])


        self.test_acc_personal = test_inference(self.args, self.model, self.testloader)
        self.test_acc_global = test_inference(self.args, global_model, self.testloader)
        self.train_global_loss = self.inference(global_model)
        self.train_personal_loss = epoch_loss[-1]

        print(f"\x1b[{32}m{'Test accuracy of personal model on Client {} is {:.2f}%'.format(UserID, 100*self.test_acc_personal)}\x1b[0m")
        print(f"\x1b[{32}m{'Test accuracy of global model on Client {} is {:.2f}%'.format(UserID, 100*self.test_acc_global)}\x1b[0m")

        return local_sum, self.test_acc_personal, self.train_personal_loss, self.test_acc_global

    def inference(self, model):
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.trainloader):
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            if self.args.model == 'MLP' or self.args.model == 'MLR':
                images = images.reshape(-1, 784)
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()
        return loss


def test_inference(args, model, testloader):
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(args.device), labels.to(args.device)
        if args.model == 'MLP' or args.model == 'MLR':
            images = images.reshape(-1, 784)
        outputs = model(images)
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy
