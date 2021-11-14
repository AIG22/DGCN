import argparse
import torch
import os

def get_citation_args():
    #os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=2e-5,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden1', type=int, default=32,
                        help='Number of hidden units.')
    #parser.add_argument('--train_rate', type=float, default=0.2,
     #                   help='Ratio of training set')
    #parser.add_argument('--apha', type=float, default=0,
    #                    help='apha set')
    parser.add_argument('--dropout', type=float, default=0.6,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dataset', type=str, default="chameleon",
                        help='Dataset to use.')
    parser.add_argument('--model', type=str, default="DGCN",
                        help='model to use.')
    parser.add_argument('--feature', type=str, default="mul",
                        choices=['mul', 'cat', 'adj'],
                        help='feature-type')
    parser.add_argument('--normalization', type=str, default='AugNormAdj',
                       choices=['AugNormAdj'],
                       help='Normalization method for the adjacency matrix.')
    #parser.add_argument('--cuda', type=int, default=2)


    args, _ = parser.parse_known_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args
