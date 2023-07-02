#test.py
#!/usr/bin/env python3

""" test neuron network performace
print top1 and top5 err on test dataset
of a model

author baiyu
"""
# 测试自己数据集下的模型准确率

import argparse

from matplotlib import pyplot as plt

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from conf import settings
from utils import get_network, get_test_dataloader, get_my_test_dataloader, get_my_train_dataloader, compute_my_mean_std

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    args = parser.parse_args()

    net = get_network(args)

    # cifar100_test_loader = get_test_dataloader(
    #     settings.CIFAR100_TRAIN_MEAN,
    #     settings.CIFAR100_TRAIN_STD,
    #     #settings.CIFAR100_PATH,
    #     num_workers=4,
    #     batch_size=args.b,
    # )
    myData_mean, myData_std = compute_my_mean_std()  # 获得样本中的std和mean
    myData_test_loader = get_my_test_dataloader(
        myData_mean,
        myData_std,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )


    net.load_state_dict(torch.load(args.weights))
    print(net)
    net.eval()

    correct_1 = 0.0  # top1
    correct_2 = 0.0  # top2
    total = 0

    with torch.no_grad():
        for n_iter, (image, label) in enumerate(myData_test_loader):
            print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(myData_test_loader)))

            if args.gpu:
                image = image.cuda()
                label = label.cuda()
                print('GPU INFO.....')
                print(torch.cuda.memory_summary(), end='')


            output = net(image)
            _, pred = output.topk(2, 1, largest=True, sorted=True)

            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()

            #compute top 2
            correct_2 += correct[:, :2].sum()

            #compute top1
            correct_1 += correct[:, :1].sum()

    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')

    print()
    print("Top 1 acc: ", correct_1 / len(myData_test_loader.dataset) * 100, "% ")
    print("Top 2 err: ", correct_2 / len(myData_test_loader.dataset) * 100, "% ")
    print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))

