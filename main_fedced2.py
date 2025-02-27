#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid
from utils.options import args_parser
from models.Update1 import LocalUpdate
from models.Nets1 import MLP, CNNMnist, CNNCifar
from models.test1 import test_img
from models.Fed import FedAvg

# 计算各个设备的参数，包括处理能力、通信延迟等
def calculate_device_parameters(data):
    device_parameters = []
    f_min = 0.1
    f_max = 1.0
    N_0 = 0.01
    kappa = 0.5
    alpha_n = 0.5

    for id, device_data in data.items():
        D_n = len(device_data)
        c_n = 10
        f_n = random.uniform(f_min, f_max)
        tau_n = random.uniform(0.1, 1.0)
        T_cmp_N1 = max([(c_n * len(data[j])) / f_max for j in data])
        T_cmp_N2 = max([(c_n * len(data[j])) / f_min for j in data])
        T_cmp_N3 = (sum([alpha_n * (c_n * len(data[j])) ** 3 for j in data]) / kappa) ** (1 / 3)
        T_cmp_star = max(T_cmp_N1, T_cmp_N2, T_cmp_N3)

        if (c_n * D_n) / f_n > T_cmp_star:
            f_n = f_max
        elif (c_n * D_n) / f_n < T_cmp_star:
            f_n = f_min
        else:
            f_n = (c_n * D_n) / T_cmp_star

        s_n = 100
        B = 10
        h_n = random.uniform(0.1, 1.0)
        p_n_min = 0.1
        p_n_max = 1.0
        tau_n_max = s_n / (B * np.log2(1 + (h_n * p_n_min) / N_0))
        tau_n_min = s_n / (B * np.log2(1 + (h_n * p_n_max) / N_0))

        if tau_n > tau_n_max:
            tau_n = tau_n_max
        elif tau_n < tau_n_min:
            tau_n = tau_n_min

        E_n_cmp = alpha_n ** 2 * c_n * D_n * f_n ** 2
        E_n_com = tau_n * (N_0 / h_n) * (np.exp(s_n / (tau_n * B)) - 1)
        SUB1 = E_n_cmp + kappa * T_cmp_star
        SUB2 = E_n_com + kappa * tau_n
        TotalCost = SUB1 + SUB2

        device_parameters.append({
            'id': id,
            'f_n': f_n,
            'tau_n': tau_n,
            'TotalCost': TotalCost
        })

    return device_parameters


# 选择总成本最低的设备
def select_optimal_devices(device_parameters, k):
    sorted_devices = sorted(device_parameters, key=lambda x: x['TotalCost'])
    selected_devices = sorted_devices[:k]
    return selected_devices


# 计算客户端信任值
def client_trust_evaluation(selected_clients, global_model, validation_loader):
    trust_scores = []
    for client in selected_clients:
        local_model = client['model']
        cs = contribution_score(local_model, global_model, validation_loader)
        rs = relevance_score(local_model, global_model)
        cf = client.get('communication_frequency', 1)
        trust_score = cs + rs + (1 - cf)
        trust_scores.append(trust_score)
    return trust_scores


# 计算本地模型的贡献分数
def contribution_score(local_model, global_model, validation_loader):
    local_model.eval()
    global_model.eval()
    criterion = nn.CrossEntropyLoss()
    local_loss = 0.0
    global_loss = 0.0

    with torch.no_grad():
        for data, target in validation_loader:
            data, target = data.to(args.device).float(), target.to(args.device)
            output_local = local_model(data)
            output_global = global_model(data)
            local_loss += criterion(output_local, target).item()
            global_loss += criterion(output_global, target).item()

    cs = (global_loss - local_loss) / max(global_loss, local_loss)
    return cs


# 计算本地模型的相关性分数
def relevance_score(local_model, global_model):
    local_params = list(local_model.parameters())
    global_params = list(global_model.parameters())
    rs = 0.0
    for local_param, global_param in zip(local_params, global_params):
        local_grad = local_param.grad
        global_grad = global_param.grad
        if local_grad is not None and global_grad is not None:
            rs += torch.sum(torch.eq(torch.sign(local_grad), torch.sign(global_grad))).item()
    rs /= len(local_params) * local_params[0].numel()
    return rs


# 聚合客户端模型
def model_aggregation(selected_clients, global_model, trust_scores):
    total_weight = sum(trust_scores)
    new_global_model = copy.deepcopy(global_model)
    new_global_model.train()

    d_i_t_values = []  # 用于记录每次 D_i_t 的值
    D_max = 10  # 初始 D_max 值，可以根据经验进行微调

    for client, trust_score in zip(selected_clients, trust_scores):
        local_model = client['model']
        local_model.eval()

        local_update = {}
        for name, param in local_model.named_parameters():
            local_update[name] = param.data - global_model.state_dict()[name]

        trust_adjusted_update = {}
        for name, update in local_update.items():
            trust_adjusted_update[name] = update * (1 + trust_score - sum(trust_scores) / len(trust_scores))

        # 计算 D_i_t
        D_i_t = 0.0
        for name in trust_adjusted_update.keys():
            update_difference = trust_adjusted_update[name] - (new_global_model.state_dict()[name] - global_model.state_dict()[name])
            D_i_t += torch.norm(update_difference) / torch.norm(global_model.state_dict()[name] - new_global_model.state_dict()[name])

        D_i_t /= len(trust_adjusted_update) if len(trust_adjusted_update) > 0 else 1  # 防止除以零
        d_i_t_values.append(D_i_t)  # 收集 D_i_t

        # 动态计算 D_max
        d_i_t_values_cpu = [d_i_t.cpu().numpy() for d_i_t in d_i_t_values]  # 将 CUDA 张量移至 CPU
        D_max = np.mean(d_i_t_values_cpu) + 2 * np.std(d_i_t_values_cpu)  # 可根据实际情况评估

        # 根据阈值判断该客户端的更新是否有效
        if D_i_t > D_max:
            continue  # 如果 D_i_t 超过阈值，跳过此客户端的更新

        # 更新全局模型参数
        for name in new_global_model.state_dict().keys():
            if name in trust_adjusted_update:
                new_global_model.state_dict()[name].data += (trust_score / total_weight) * trust_adjusted_update[name]

    return new_global_model  # 返回新的全局模型


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print(f"Using device: {args.device}")

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)

        # 输出数据集信息
        print(f'Training data shape: {dataset_train.data.shape}')
        print(f'Testing data shape: {dataset_test.data.shape}')

        if dataset_train.data.ndimension == 3:
            dataset_train.data = dataset_train.data.unsqueeze(1)  # [num_samples, 1, 28, 28]
        if dataset_test.data.ndimension == 3:
            dataset_test.data = dataset_test.data.unsqueeze(1)  # [num_samples, 1, 28, 28]

        print(f'Updated training data shape: {dataset_train.data.shape}')
        print(f'Updated testing data shape: {dataset_test.data.shape}')

        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)

    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)

        print(f'Training data shape: {dataset_train.data.shape}')
        print(f'Testing data shape: {dataset_test.data.shape}')

        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            dict_users = cifar_noniid(dataset_train, args.num_users)
    else:
        exit('Error: unrecognized dataset')

    # 创建验证数据加载器
    validation_loader = DataLoader(dataset_test, batch_size=10, shuffle=False)

    # 输出图像大小
    img_size = dataset_train.data[0].shape  # [C, H, W]
    print(f"Image size: {img_size}")

    # 初始化全局模型
    if args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')

    print("Initialized global model:")
    print(net_glob)

    net_glob.train()

    loss_train = []
    for iter in range(args.epochs):
        print(f"Round {iter + 1}/{args.epochs}")
        loss_locals = []
        m = min(100, args.num_users)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        devices = [{'id': idx, 'model': copy.deepcopy(net_glob)} for idx in idxs_users]

        # 本地模型训练
        for device in devices:
            idx = device['id']
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=device['model'].to(args.device))  # 在GPU上训练
            device['model'].load_state_dict(w)  # 加载训练后的权重
            loss_locals.append(loss)

        # 计算设备参数并选择最优设备
        data = {device['id']: dict_users[device['id']] for device in devices}
        device_parameters = calculate_device_parameters(data)
        selected_devices = select_optimal_devices(device_parameters, k=m)

        selected_devices_with_models = [{'id': device['id'], 'model': device['model']} for device in devices if
                                        device['id'] in [selected['id'] for selected in selected_devices]]

        # 评估信任分数
        trust_scores = client_trust_evaluation(selected_devices_with_models, net_glob, validation_loader)
        new_global_model = model_aggregation(selected_devices_with_models, net_glob, trust_scores)

        # 更新全局模型
        net_glob.load_state_dict(new_global_model.state_dict())

        loss_avg = sum(loss_locals) / len(loss_locals)
        print(f'Round {iter + 1}, Average loss {loss_avg:.3f}')
        loss_train.append(loss_avg)

        # 测试模型并输出准确率
        net_glob.eval()
        acc_train, train_loss = test_img(net_glob.to(args.device), dataset_train, args)
        acc_test, loss_test = test_img(net_glob.to(args.device), dataset_test, args)
        print("Training accuracy: {:.2f}".format(acc_train))
        print("Testing accuracy: {:.2f}".format(acc_test))

    # 绘制训练损失图
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.xlabel('Round')
    plt.title('Training Loss Over Time')
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))
    print("Training loss plot saved.")
