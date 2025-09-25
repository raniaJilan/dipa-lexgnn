import os
import argparse
import torch
import random
import numpy as np
from model import LEXGNN
from data_handler import load_data
from model_trainer import *
import warnings
warnings.filterwarnings("ignore")


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed) 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='yelp', help='Dataset to use: amazon or yelp')
    parser.add_argument('--seed', type=int, default=2, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--train_ratio', type=float, default=0.4, help='Ratio of training data')
    parser.add_argument('--test_ratio', type=float, default=0.67, help='Ratio of test data')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
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
    print()
    print('Dataset:', args.data_name.upper())

    # Set seed
    set_random_seed(args.seed)

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
    auc, f1, gm, ap, auc1, prec, rec = test(model_best, test_loader, device)
    print('===========================================')
    print(f'AUC_cls: {auc:.4f} | AUC_pre: {auc1:.4f} | F1-macro: {f1:.4f} | P-macro: {prec:.4f} | R-macro: {rec:.4f} | G-mean: {gm:.4f} | AP: {ap:.4f} | Epoch-time: {et:.4f}') 
    print()  


if __name__ == '__main__':
    main()
