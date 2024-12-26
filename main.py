import os
import sys
import time
import copy

import numpy as np
import math
import random

import functools
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils

import wandb

import pickle

import datasets
import models
import init
import training

import measures

def run( args):

    # reduce batch_size when larger than train_size
    if (args.batch_size >= args.train_size):
        args.batch_size = args.train_size
    
    assert (args.train_size%args.batch_size)==0, 'batch_size must divide train_size!'

    if args.accumulation:
        accumulation = args.train_size // args.batch_size
    else:
        accumulation = 1

    train_loader, test_loader = init.init_data( args)

    model = init.init_model( args)
    model0 = copy.deepcopy( model)

    if args.scheduler_time is None:
        args.scheduler_time = args.max_epochs
    criterion, optimizer, scheduler = init.init_training( model, args)

    dynamics, best = init.init_output( model, criterion, train_loader, test_loader, args)
    if args.print_freq >= 10:
        print_ckpts = init.init_loglinckpt( args.print_freq, args.max_epochs, fill=True)
    else:
        print_ckpts = init.init_loglinckpt( args.print_freq, args.max_epochs, fill=False)
    save_ckpts = init.init_loglinckpt( args.save_freq, args.max_epochs, fill=False)

    print_ckpt = next(print_ckpts)
    save_ckpt = next(save_ckpts)

    start_time = time.time()

    test_loss, test_acc, avg_epoch_time = best['loss'], best['acc'], None # for first few wandb loga

    for epoch in range(args.max_epochs):

        loss = training.train( model, train_loader, accumulation, criterion, optimizer, scheduler)

        # Log current & best dynamics to wandb
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': loss,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'lr': scheduler.get_last_lr()[0],
            'avg_epoch_time': avg_epoch_time,
            'best_epoch': best['epoch'],
            'best_loss': best['loss'],
            'best_acc': best['acc']
        })

        if (epoch+1)==print_ckpt or (epoch+1 == args.max_epochs):

            avg_epoch_time = (time.time()-start_time)/(epoch+1)
            test_loss, test_acc = measures.test(model, test_loader)

            if test_loss<best['loss']: # update best model if loss is smaller
                best['epoch'] = epoch+1
                best['loss'] = test_loss
                best['acc'] = test_acc
                best['model'] = copy.deepcopy( model.state_dict())

            dynamics.append({'t': epoch+1, 'trainloss': loss, 'testloss': test_loss, 'testacc': test_acc})
            print(
            'Epoch : ', epoch + 1,
            '\t train loss: {:06.4f}'.format(loss),
            ', test loss: {:06.4f}'.format(test_loss),
            ', test acc.: {:04.2f}'.format(test_acc),
            ', lr: {:.5e}'.format(scheduler.get_last_lr()[0]),
            ', epoch time: {:06.2f}'.format(avg_epoch_time)
            )

            if epoch+1 == args.max_epochs: print('This was the max epoch!')
            print_ckpt = next(print_ckpts)

        if loss <= args.loss_threshold:

            print(f'Loss={loss} passed threshold={args.loss_threshold}, this is last epoch ...')
            output = {
                'init': model0.state_dict(),
                'best': best,
                'model': copy.deepcopy(model.state_dict()),
                'dynamics': dynamics,
                'epoch': epoch+1
            }
            with open(args.outname, "wb") as handle:
                pickle.dump(args, handle)
                pickle.dump(output, handle)

            break
        elif (epoch+1)==save_ckpt:

            print(f'Checkpoint at epoch {epoch+1}, saving data ...')
            output = {
                'init': model0.state_dict(),
                'best': best,
                'model': copy.deepcopy(model.state_dict()),
                'dynamics': dynamics,
                'epoch': epoch+1
            }
            with open(args.outname, "wb") as handle:
                pickle.dump(args, handle)
                pickle.dump(output, handle)
            save_ckpt = next(save_ckpts)


    print(f"Training time: {(time.time() - start_time):.2f}")
    print('Best test stats:')
    print(f"  Best epoch   : {best['epoch']}")
    print(f"  Best loss    : {best['loss']:.4f}")
    print(f"  Best accuracy: {best['acc']:.2f}%")

    return None

torch.set_default_dtype(torch.float32)

parser = argparse.ArgumentParser(description='Supervised Learning of the Random Hierarchy Model with deep neural networks')
parser.add_argument("--device", type=str, default='cuda')
'''
	DATASET ARGS
'''
parser.add_argument('--dataset', default="rhm", type=str)
parser.add_argument('--mode', type=str, required=True, choices=['masked', 'AR_last', 'AR_window'])
parser.add_argument('--num_features', metavar='v', type=int, help='number of features')
parser.add_argument('--num_classes', metavar='n', type=int, help='number of classes')
parser.add_argument('--num_synonyms', metavar='m', type=int, help='multiplicity of low-level representations')
parser.add_argument('--tuple_size', metavar='s', type=int, help='size of low-level representations')
parser.add_argument('--num_layers', metavar='L', type=int, help='number of layers')
parser.add_argument("--seed_rules", type=int, help='seed for the dataset')
parser.add_argument("--num_tokens", type=int, help='number of input tokens (spatial size)')
parser.add_argument('--train_size', metavar='Ptr', type=int, help='training set size')
parser.add_argument('--batch_size', metavar='B', type=int, help='batch size')
parser.add_argument('--test_size', metavar='Pte', type=int, help='test set size')
parser.add_argument("--seed_sample", type=int, help='seed for the sampling of train and testset')
parser.add_argument("--replacement", default=False, action="store_true", help='allow for replacement in the dataset sampling')
parser.add_argument('--input_format', type=str, default='onehot')
parser.add_argument('--whitening', type=int, default=0)
'''
	ARCHITECTURE ARGS
'''
parser.add_argument('--model', type=str, help='architecture (fcn and hcnn implemented)', default='transformer_mla', choices=['fcn', 'hcnnn', 'hlcn', 'transformer_mla'])
parser.add_argument('--depth', type=int, help='depth of the network')
parser.add_argument('--width', type=int, help='width of the network')
parser.add_argument("--filter_size", type=int, default=None)
parser.add_argument('--num_heads', type=int, help='number of heads (transformer only)')
parser.add_argument('--embedding_dim', type=int, help='embedding dimension (transformer only)')
parser.add_argument('--bias', default=False, action='store_true')
parser.add_argument("--seed_model", type=int, help='seed for model initialization')
'''
       TRAINING ARGS
'''
parser.add_argument('--lr', type=float, help='learning rate', default=0.1)
parser.add_argument('--optim', type=str, default='sgd')
parser.add_argument('--accumulation', default=False, action='store_true')
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--scheduler', type=str, default=None)
parser.add_argument('--scheduler_time', type=int, default=None)
parser.add_argument('--max_epochs', type=int, default=100)
'''
	OUTPUT ARGS
'''
parser.add_argument('--print_freq', type=int, help='frequency of prints', default=10)
parser.add_argument('--save_freq', type=int, help='frequency of saves', default=10)
parser.add_argument('--loss_threshold', type=float, default=1e-3)
parser.add_argument('--outname', type=str, required=True, help='path of the output file')

args = parser.parse_args()

group = f'{os.environ.get("SLURM_ARRAY_JOB_ID", "unknown")}-{os.environ.get("SLURM_JOB_NAME", "unknown")}'

wandb.init(
    project="Random Hierarchy Model",
    entity='tate-ubc',
    config=args,
    group=group,
    name=f'({os.environ.get("SLURM_ARRAY_TASK_ID", "unknown")}) {os.path.basename(args.outname)}'
)
run( args)
