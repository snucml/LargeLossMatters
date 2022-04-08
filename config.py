import os
import sys
import argparse
from munch import Munch as mch
from os.path import join as ospj


_DATASET = ('pascal', 'coco', 'nuswide', 'cub')
_TRAIN_SET_VARIANT = ('observed', 'clean')
_OPTIMIZER = ('adam', 'sgd')
_SCHEMES = ('LL-R', 'LL-Ct', 'LL-Cp')
_LOOKUP = {
    'feat_dim': {
        'resnet50': 2048
    },
    'num_classes': {
        'pascal': 20,
        'coco': 80,
        'nuswide': 81,
        'cub': 312
    }
}



def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def set_dir(runs_dir, exp_name):
    runs_dir = ospj(runs_dir, exp_name)
    if not os.path.exists(runs_dir):
        os.makedirs(runs_dir)
    return runs_dir



def set_follow_up_configs(args):
    args.feat_dim = _LOOKUP['feat_dim'][args.arch]
    args.num_classes = _LOOKUP['num_classes'][args.dataset]
    args.save_path = set_dir(args.save_path, args.exp_name)

    if args.mod_scheme == 'LL-R':
        args.llr_rel = 1
        args.llr_rel_mod = 0
        args.perm_mod = 0
    elif args.mod_scheme == 'LL-Ct':
        args.llr_rel = 1
        args.llr_rel_mod = 1
        args.perm_mod = 0
    elif args.mod_scheme == 'LL-Cp':
        args.llr_rel = 1
        args.llr_rel_mod = 1
        args.perm_mod = 1
    
    if args.delta_rel != 0:
        args.delta_rel /= 100

    return args


def get_configs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--ss_seed', type=int, default=999,
                        help='seed fo subsampling')
    parser.add_argument('--ss_frac_train', type=float, default=1.0,
                        help='fraction of training set to subsample')
    parser.add_argument('--ss_frac_val', type=float, default=1.0,
                        help='fraction of val set to subsample')
    parser.add_argument('--use_feats', type=str2bool, nargs='?',
                        const=True, default=False,
                        help='False if end-to-end training, True if linear training')
    parser.add_argument('--val_frac', type=float, default=0.2)
    parser.add_argument('--split_seed', type=int, default=1200)
    parser.add_argument('--train_set_variant', type=str, default='observed',
                        choices=_TRAIN_SET_VARIANT)
    parser.add_argument('--val_set_variant', type=str, default='clean')
    parser.add_argument('--arch', type=str, default='resnet50')
    parser.add_argument('--freeze_feature_extractor', type=str2bool, nargs='?',
                        const=True, default=False)
    parser.add_argument('--use_pretrained', type=str2bool, nargs='?',
                        const=True, default=True)
    
    
    parser.add_argument('--save_path', type=str, default='./results')
    parser.add_argument('--exp_name', type=str, default='exp_default')
    parser.add_argument('--dataset', type=str, default='coco',
                        choices=_DATASET)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu_num', type=str, default='0')
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=_OPTIMIZER)
    parser.add_argument('--bsize', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--lr_mult', type=float, default=10)
    parser.add_argument('--mod_scheme', type=str, default='LL-R', 
                        choices=_SCHEMES)
    parser.add_argument('--delta_rel', type=float, default=0)
    
    args = parser.parse_args()
    args = set_follow_up_configs(args)
    args = mch(**vars(args))

    return args


