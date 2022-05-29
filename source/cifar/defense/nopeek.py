import os
import sys
import argparse
import random
import shutil
import numpy as np
import torch
import torch.optim as optim

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_DIR, '../'))
sys.path.append(os.path.join(FILE_DIR, '../../'))
SAVE_ROOT = os.path.join(FILE_DIR, '../../../results/%s/%s/nopeek')
from base import CIFARTrainer
import models as models
import torch.nn as nn
import time
from utils import mkdir, str2bool, load_yaml, write_yaml, adjust_learning_rate, plot_hist
from utils import mkdir, str2bool, write_yaml, load_yaml, adjust_learning_rate, \
    AverageMeter, Bar, plot_hist, accuracy, one_hot_embedding, CrossEntropy_soft

#############################################################################################################
# get and save the arguments
#############################################################################################################
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', '-name', type=str, help='experiment name, used for set up save_dir')
    parser.add_argument('--dataset', type=str, choices=['CIFAR10', 'CIFAR100'], help='dataset name')
    parser.add_argument('--random_seed', '-s', type=int, default=1000, help='random seed')
    parser.add_argument('--model', type=str, help='model architecture')
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--schedule_milestone', type=int, nargs='+', help='when to decrease the learning rate')
    parser.add_argument('--gamma', type=float, help='learning rate step gamma')
    parser.add_argument('--weight_decay', type=float, help='weight decay')
    parser.add_argument('--momentum', type=float, help='momentum')
    parser.add_argument('--train_batchsize', type=int, help='training batch size')
    parser.add_argument('--test_batchsize', type=int, help='testing batch size')
    parser.add_argument('--num_workers', type=int, help='number of workers')
    parser.add_argument('--num_epochs', '-ep', type=int, help='number of epochs')
    parser.add_argument('--partition', type=str, choices=['target', 'shadow'], help='training partition')
    parser.add_argument('--if_resume', type=str2bool, help='If resume from checkpoint')
    parser.add_argument('--if_data_augmentation', '-aug', type=str2bool, help='If use data augmentation')
    parser.add_argument('--if_onlyeval', type=str2bool, help='If only evaluate the pre-trained model')
    parser.add_argument('--npv', type=float, default=0.0)
    return parser


def check_args(parser):
    """Check and store the arguments as well as set up the save_dir"""
    ## set up save_dir
    args = parser.parse_args()
    save_dir = os.path.join(SAVE_ROOT % (args.dataset, args.model), args.exp_name)
    if args.partition == 'shadow':
        save_dir = os.path.join(save_dir, 'shadow')
    mkdir(save_dir)
    if args.random_seed is None:
        args.random_seed = random.randint(1, 10 ^ 5)

    ## load configs and store the parameters
    if args.if_onlyeval:
        preload_configs = load_yaml(os.path.join(save_dir, 'params.yml'))
        parser.set_defaults(**preload_configs)
        args = parser.parse_args()
    else:
        default_configs = load_yaml(FILE_DIR + '/configs/default.yml')
        parser.set_defaults(**default_configs)
        args = parser.parse_args()
        write_yaml(vars(args), os.path.join(save_dir, 'params.yml'))

    ## store this script
    shutil.copy(os.path.realpath(__file__), save_dir)
    return args, save_dir


#############################################################################################################
# main function
#############################################################################################################
class NoPeekLoss(torch.nn.modules.loss._Loss):
    def __init__(self, dcor_weighting: float = 0.1) -> None:
        super().__init__()
        self.dcor_weighting = dcor_weighting

        self.ce = torch.nn.CrossEntropyLoss()
        self.dcor = DistanceCorrelationLoss()

    def forward(self, inputs, intermediates, outputs, targets):
        cce_loss = self.ce(outputs, targets.long())

        # DistanceCorrelationLoss is costly, so only calc it if necessary
        if self.dcor_weighting > 0.0:
            dcor_loss = self.dcor_weighting * self.dcor(inputs, intermediates)
            return cce_loss, dcor_loss
        else:
            return (cce_loss,)


class DistanceCorrelationLoss(torch.nn.modules.loss._Loss):
    def forward(self, input_data, intermediate_data):
        input_data = input_data.view(input_data.size(0), -1)
        intermediate_data = intermediate_data.view(intermediate_data.size(0), -1)

        # Get A matrices of data
        A_input = self._A_matrix(input_data)
        A_intermediate = self._A_matrix(intermediate_data)

        # Get distance variances
        input_dvar = self._distance_variance(A_input)
        intermediate_dvar = self._distance_variance(A_intermediate)

        # Get distance covariance
        dcov = self._distance_covariance(A_input, A_intermediate)

        # Put it together
        dcorr = dcov / (input_dvar * intermediate_dvar).sqrt()

        return dcorr

    def _distance_covariance(self, a_matrix, b_matrix):
        return (a_matrix * b_matrix).sum().sqrt() / a_matrix.size(0)

    def _distance_variance(self, a_matrix):
        return (a_matrix ** 2).sum().sqrt() / a_matrix.size(0)

    def _A_matrix(self, data):
        distance_matrix = self._distance_matrix(data)

        row_mean = distance_matrix.mean(dim=0, keepdim=True)
        col_mean = distance_matrix.mean(dim=1, keepdim=True)
        data_mean = distance_matrix.mean()

        return distance_matrix - row_mean - col_mean + data_mean

    def _distance_matrix(self, data):
        n = data.size(0)
        distance_matrix = torch.zeros((n, n))

        for i in range(n):
            for j in range(n):
                row_diff = data[i] - data[j]
                distance_matrix[i, j] = (row_diff ** 2).sum()

        return distance_matrix

class Trainer(CIFARTrainer):
    def set_criterion(self):
        """Set up the relaxloss training criterion"""
        self.criterion = nn.CrossEntropyLoss()
        self.crossentropy = nn.CrossEntropyLoss()
        self.crossentropy_noreduce = nn.CrossEntropyLoss(reduction='none')
        self.nploss = NoPeekLoss(self.npv)

    def train(self, model, optimizer):
        """Train"""
        model.train()
        criterion = self.criterion
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        batch_time = AverageMeter()
        dataload_time = AverageMeter()
        time_stamp = time.time()

        bar = Bar('Processing', max=len(self.trainloader))
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            ### Record the data loading time
            dataload_time.update(time.time() - time_stamp)

            ### Output
            outputs, intermediates = model(inputs)
            # loss = criterion(outputs, targets.long()) + model.loss()

            loss = self.nploss(inputs, intermediates, outputs, targets)
            ### Record accuracy and loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            ### Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ### Record the total time for processing the batch
            batch_time.update(time.time() - time_stamp)
            time_stamp = time.time()

            ### Progress bar
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                batch=batch_idx + 1,
                size=len(self.trainloader),
                data=dataload_time.avg,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg,
            )
            bar.next()

        bar.finish()
        return (losses.avg, top1.avg, top5.avg)

def main():
    args, save_dir = check_args(parse_arguments())

    ### Set up trainer and model
    trainer = Trainer(args, save_dir)
    trainer.npv = args.npv
    model = models.__dict__[args.model](num_classes=trainer.num_classes)
    # model = torch.nn.DataParallel(model)
    model.to(trainer.device)
    torch.backends.cudnn.benchmark = True
    print('Total params: %.2f' % (sum(p.numel() for p in model.parameters())))
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    ### Load checkpoint
    start_epoch = 0
    logger = trainer.logger
    if args.if_resume or args.if_onlyeval:
        try:
            print('==> Resuming from checkpoint..')
            checkpoint = torch.load(os.path.join(save_dir, 'checkpoint.pkl'))
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['opt_state_dict'])
            logger = Logger(os.path.join(save_dir, 'log.txt'), title=title, resume=True)
        except:
            pass

    if args.if_onlyeval:
        print('\nEvaluation only')
        test_loss, test_acc, test_acc5 = trainer.test(model)
        print(' Test Loss: %.8f, Test Acc(top1): %.2f, Test Acc(top5): %.2f' % (test_loss, test_acc, test_acc5))
        return

    ### Training
    for epoch in range(start_epoch, args.num_epochs):
        adjust_learning_rate(optimizer, epoch, args.gamma, args.schedule_milestone)
        train_loss, train_acc, train_acc5 = trainer.train(model, optimizer)
        test_loss, test_acc, test_acc5 = trainer.test(model)
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        logger.append([lr, train_loss, test_loss, train_acc, test_acc, train_acc5, test_acc5])
        print('Epoch %d, Train acc: %f, Test acc: %f, lr: %f' % (epoch, train_acc, test_acc, lr))

        ### Save checkpoint
        save_dict = {'epoch': epoch,
                     'model_state_dict': model.state_dict(),
                     'opt_state_dict': optimizer.state_dict()}
        torch.save(save_dict, os.path.join(save_dir, 'checkpoint.pkl'))
        torch.save(model, os.path.join(save_dir, 'model.pt'))

    ### Visualize
    trainer.logger_plot()
    train_losses, test_losses = trainer.get_loss_distributions(model)
    plot_hist([train_losses, test_losses], ['train', 'val'], os.path.join(save_dir, 'hist_ep%d.png' % epoch))


if __name__ == '__main__':
    main()
