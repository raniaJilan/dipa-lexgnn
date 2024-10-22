import os
import argparse
import torch
import random
import numpy as np
from model import LEXGNN
from data_handler import load_data
from model_handler import *


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='yelp', help='Dataset to use: amazon or yelp')
    parser.add_argument('--seed', type=int, default=2, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--train_ratio', type=float, default=0.4, help='Ratio of training data')
    parser.add_argument('--test_ratio', type=float, default=0.67, help='Ratio of test data')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--valid_epochs', type=int, default=3)
    parser.add_argument('--early_stop', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--wd', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--cuda_id', type=int, default=0)
    parser.add_argument('--n_layer', type=int, default=2)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--n_hidden', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--beta', type=float, default=0.5)
    
    args = parser.parse_args()

    # GPU
    device = torch.device(args.cuda_id)
    torch.cuda.set_device(device)

    # Load data
    n_input, train_loader, valid_loader, test_loader = load_data(
                            args.data_name, args.seed, args.train_ratio, args.test_ratio, args.n_layer, args.batch_size)

    # Define model
    model = LEXGNN(n_input, 2, args.n_hidden, args.n_layer, args.n_head, args.dropout).to(device)

    # Train 
    model_best, ep, et = train(model, train_loader, valid_loader, args.epochs, args.valid_epochs, 
                                 args.beta, args.lr, args.wd, args.early_stop, args.seed, device)

    # Test 
    auc, f1, gm, auc1 = test(model_best, test_loader)
    print('-------------------------------------------------------------------------------------')
    print('AUC_cls:', round(auc,4), ' AUC_pre:', round(auc1,4), ' F1_mac:', round(f1,4), ' G-mean:', round(gm,4))    
    


if __name__ == '__main__':
    main()
