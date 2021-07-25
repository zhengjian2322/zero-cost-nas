# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import argparse
import pickle
import time
import torch
import torch.optim as optim
import torch.nn.functional as F

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.contrib.handlers import ProgressBar

from foresight.models import nasbench2
from foresight.pruners import predictive
from foresight.dataset import *
from config_space import get_benchmark201_config_space, get_darts_config_space
from model.darts.model import DartsNetworkCIFAR
from utils import config_to_genotype


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train models')
    parser.add_argument('--api_loc', default='data/NAS-Bench-201-v1_0-e61699.pth',
                        type=str, help='path to API')
    parser.add_argument('--outdir', default='./',
                        type=str, help='output directory')
    parser.add_argument('--outfname', default='test',
                        type=str, help='output filename')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--init_channels', default=16, type=int)
    parser.add_argument('--img_size', default=8, type=int)
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset to use [cifar10, cifar100, ImageNet16-120]')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to work on')
    parser.add_argument('--num_data_workers', type=int, default=2, help='number of workers for dataloaders')
    parser.add_argument('--dataload', type=str, default='random', help='random or grasp supported')
    parser.add_argument('--dataload_info', type=int, default=1,
                        help='number of batches to use for random dataload or number of samples per class for grasp dataload')
    parser.add_argument('--write_freq', type=int, default=5, help='frequency of write to file')
    parser.add_argument('--logmeasures', action="store_true", default=False,
                        help='add extra logging for predictive measures')
    parser.add_argument('--arch_name',  default='darts',
                        help='architecture name in [darts, benchmark201]')
    args = parser.parse_args()
    args.device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    return args


def get_net_from_config(net_name, config, n_class, init_channels):
    if net_name == 'benchmark201':
        config = '|%s~0|+|%s~0|%s~1|+|%s~0|%s~1|%s~2|' % (config['op_0'],
                                                          config['op_1'], config['op_2'],
                                                          config['op_3'], config['op_4'], config['op_5'])
        net = nasbench2.get_model_from_arch_str(config, n_class, init_channels=init_channels)
    elif net_name == 'darts':
        genotype = config_to_genotype(config)
        net = DartsNetworkCIFAR(C=init_channels, num_classes=n_class,
                                layers=20, auxiliary=False, genotype=genotype)
        net.drop_path_prob = 0
    else:
        raise ValueError('%s is not supported' % net_name)
    return net


def get_num_classes(dataset):
    return 100 if dataset == 'cifar100' else 10 if dataset == 'cifar10' else 120


def setup_experiment(net, args):
    optimizer = optim.SGD(
        net.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=0.0005,
        nesterov=True)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0, last_epoch=-1)

    train_loader, val_loader = get_cifar_dataloaders(args.batch_size, args.batch_size, args.dataset,
                                                     args.num_data_workers, resize=args.img_size)

    return optimizer, lr_scheduler, train_loader, val_loader


def load_arch_config_space(arch_name='benchmark201'):
    if arch_name == 'benchmark201':
        return get_benchmark201_config_space()
    elif arch_name == 'darts':
        return get_darts_config_space()
    else:
        raise ValueError('%s is not supported' % arch_name)


def train():
    args = parse_arguments()
    # archs = pickle.load(open(args.api_loc, 'rb'))
    archs_config_space = load_arch_config_space(args.arch_name)

    pre = 'cf' if 'cifar' in args.dataset else 'im'
    if args.outfname == 'test':
        fn = f'darts_train_{pre}{get_num_classes(args)}_r{args.img_size}_c{args.init_channels}_e{args.epochs}.p'
    else:
        fn = f'{args.outfname}.p'
    output_file = os.path.join(args.outdir, fn)
    print('outfile =', output_file)

    cached_res = []
    archs = archs_config_space.sample_configuration(50)
    for i, config in enumerate(archs):

        num_class = get_num_classes(args.dataset)
        res = {'idx': i, 'config': config}

        net = get_net_from_config(args.arch_name, config, num_class, args.init_channels)
        net.to(args.device)

        optimizer, lr_scheduler, train_loader, val_loader = setup_experiment(net, args)

        criterion = F.cross_entropy
        trainer = create_supervised_trainer(net, optimizer, criterion, args.device)
        evaluator = create_supervised_evaluator(net, metrics={
            'accuracy': Accuracy(),
            'loss': Loss(criterion)
        })

        pbar = ProgressBar()
        pbar.attach(trainer)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log(engine):
            lr_scheduler.step()

            evaluator.run(val_loader)

            metrics = evaluator.state.metrics
            avg_accuracy = metrics['accuracy']
            avg_loss = metrics['loss']

            pbar.log_message(
                f"Validation Results - Epoch: {engine.state.epoch} Avg accuracy: {round(avg_accuracy * 100, 2)}% Val loss: {round(avg_loss, 2)} Train loss: {round(engine.state.output, 2)}")

            measures = {}
            if args.logmeasures:
                measures = predictive.find_measures(net,
                                                    train_loader,
                                                    (args.dataload, args.dataload_info, get_num_classes(args)),
                                                    args.device)
            measures['train_acc'] = engine.state.output
            measures['val_acc'] = avg_accuracy
            measures['loss'] = avg_loss
            measures['epoch'] = engine.state.epoch
            res['log_measures'].append(measures)

        evaluator.run(val_loader)

        measures = {}
        if args.logmeasures:
            measures = predictive.find_measures(net,
                                                train_loader,
                                                (args.dataload, args.dataload_info, get_num_classes(args)),
                                                args.device)
        measures['train_acc'] = 0
        measures['val_acc'] = evaluator.state.metrics['accuracy']
        measures['loss'] = evaluator.state.metrics['loss']
        measures['epoch'] = 0
        res['log_measures'].append(measures)

        stime = time.time()
        trainer.run(train_loader, args.epochs)
        etime = time.time()

        res['time'] = etime - stime

        cached_res.append(res)
        if i % args.write_freq == 0:
            print(f'writing {len(cached_res)} results to {output_file}')
            with open(output_file, 'ab') as pf:
                for cr in cached_res:
                    pickle.dump(cr, pf)
            cached_res = []


if __name__ == '__main__':
    train()
