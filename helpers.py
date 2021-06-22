import os, random, inspect
import math
import time
import torch
import pickle
import numpy as np
from scipy import sparse
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torchvision import datasets, transforms
from bisect import bisect_right, bisect_left
from numpy.lib.stride_tricks import as_strided
from sklearn.model_selection import train_test_split
from sklearn import datasets as skdsets
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from functools import partial

from optimizers import CGQ, Sls, Lookahead, COCOB_Backprop,PalOptimizer

def get_dataset(name='cifar100', target_type=torch.FloatTensor, target_shape=None, model='densenet', test_size=0.2):
    root = '/data'
    train_set, test_set = None, None
    if 'cifar' in name:
        # for CIFAR dataset
        if 'densenet' in model:
            mean = [0.5071, 0.4867, 0.4408]
            stdv = [0.2675, 0.2565, 0.2761]
        elif 'vgg' in model:
            mean=[0.485, 0.456, 0.406]
            stdv=[0.229, 0.224, 0.225]
        else:
            mean=[0.4914, 0.4822, 0.4465]
            stdv=[0.2023, 0.1994, 0.2010]
        trans_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=stdv),
        ])
        trans_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=stdv),
        ])
        if name == 'cifar100':
            train_set = datasets.CIFAR100(root=root, train=True, transform=trans_train, download=True)
            test_set = datasets.CIFAR100(root=root, train=False, transform=trans_test, download=False)
        elif name == 'cifar10':
            train_set = datasets.CIFAR10(root=root, train=True, transform=trans_train, download=True)
            test_set = datasets.CIFAR10(root=root, train=False, transform=trans_test, download=False)
        else:
            raise Exception('dataset not recognized.')
    elif name=='mnist':
        root = '../data'
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
        # if not exist, download mnist dataset
        train_set = datasets.MNIST(root=root, train=True, transform=trans, download=True)
        test_set = datasets.MNIST(root=root, train=False, transform=trans, download=True)
    elif name=='svhn':
        trans = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
        # if not exist, download mnist dataset
        temp_root = '../data/'
        if not os.path.isdir(temp_root):
            os.mkdir(temp_root)
        train_set = datasets.SVHN(root=temp_root, split='train', transform=trans, download=True)
        test_set = datasets.SVHN(root=temp_root, split='test', transform=trans, download=True)

    elif name=='digits':
        data, target = load_digits(n_class=10, return_X_y=True)
        if target_shape:
            target = target.reshape(*target_shape)
        X_train, X_test, y_train, y_test = train_test_split(data,target,test_size=0.2,random_state=101)
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        train_set = torch.utils.data.TensorDataset(torch.FloatTensor(X_train), target_type(y_train))
        test_set = torch.utils.data.TensorDataset(torch.FloatTensor(X_test), target_type(y_test))

    elif name=='boston':
        data, target = load_boston(return_X_y=True)
        if target_shape:
            target = target.reshape(*target_shape)
        X_train, X_test, y_train, y_test = train_test_split(data,target,test_size=0.2,random_state=101)
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        train_set = torch.utils.data.TensorDataset(torch.FloatTensor(X_train), target_type(y_train))
        test_set = torch.utils.data.TensorDataset(torch.FloatTensor(X_test), target_type(y_test))
    elif name=='imagenet':
        # The ImageNet Large Scale Visual Recognition Challenge (ILSVRC) dataset has 1000 categories 
        # and 1.2 million images. The images do not need to be preprocessed or packaged in any 
        # database, but the validation images need to be moved into appropriate subfolders.

        ##### Download the images from http://image-net.org/download-images
        ##### Extract the training data:
        # mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
        # tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
        # find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
        # cd ..
        ##### Extract the validation data and move images to subfolders:
        # mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
        # wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
        root += '/ILSVRC2012'
        folders = {x:os.path.join(root, x) for x in ['train','val']}
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]),
            'val': transforms.Compose([
                # transforms.Scale(256),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]),
        }
        
        # Create training and validation datasets
        image_datasets = {x: datasets.ImageFolder(folders[x], data_transforms[x]) for x in ['train', 'val']}
        train_set,test_set = image_datasets['train'], image_datasets['val']
    elif name=='tinyimagenet':
        # The ImageNet Large Scale Visual Recognition Challenge (ILSVRC) dataset has 1000 categories 
        # and 1.2 million images. The images do not need to be preprocessed or packaged in any 
        # database, but the validation images need to be moved into appropriate subfolders.

        ##### Download the images from http://image-net.org/download-images
        ##### Extract the training data:
        # mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
        # tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
        # find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
        # cd ..
        ##### Extract the validation data and move images to subfolders:
        # mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
        # wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
        root += '/tiny-imagenet-200'
        folders = {x:os.path.join(root, x) for x in ['train','val']}
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        data_transforms = {
            'train': transforms.Compose([
                # transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]),
            'val': transforms.Compose([
                # transforms.Resize(256),
                # transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]),
        }
        
        # Create training and validation datasets
        image_datasets = {x: datasets.ImageFolder(folders[x], data_transforms[x]) for x in ['train', 'val']}
        train_set,test_set = image_datasets['train'], image_datasets['val']
            
    return train_set, test_set


def get_optimizer(model, hyper={}, epochs=None, solver='sgd', lookahead=False, dp=False, dp_steps=2):
    """This is where users choose their optimizer and define the
       hyperparameter space they'd like to search."""
    lr = hyper.setdefault('lr', 0.1)
    momentum = hyper.setdefault('momentum', 0.9)
    wd = hyper.setdefault('wd',1e-4)
    nesterov = hyper.setdefault('nesterov',momentum>0)
    polak_ribiere = hyper.setdefault('polak_ribiere',False)
    mbound = hyper.setdefault('mbound',0.9)
    interp = hyper.setdefault('interp', '2pt')
    if solver=='adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=wd
        )
        scheduler = None
    elif solver == 'adagrad':
        optimizer = torch.optim.Adagrad(
            model.parameters(), lr=lr, weight_decay=wd
        )
        scheduler = None
    elif solver == 'armijo':
        optimizer = Sls(
            model.parameters(),n_batches_per_epoch=hyper['n_batches_per_epoch'],
            init_step_size=hyper['init_step_size'],
        )
        scheduler = None
    elif solver == 'cgq':
        optimizer = CGQ(
            model.parameters(), lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=wd,
            est_bound=hyper['est_bound'], 
            est_step_size=hyper['est_step_size'], 
            est_window=hyper['est_window'],
            mbound=hyper['mbound'],
            polak_ribiere=hyper['polak_ribiere'],
            interp=hyper['interp'],
            ls_prob=hyper['ls_prob'],
        )
        scheduler = None
    elif solver == 'adamq':
        optimizer = AdamQ(
            model.parameters(), lr=lr, weight_decay=wd,
            est_bound=hyper['est_bound'], 
            est_step_size=hyper['est_step_size'], 
            est_window=hyper['est_window'],
            mbound=hyper['mbound'],
            polak_ribiere=hyper['polak_ribiere'],
            interp=hyper['interp'] 
        )
        scheduler = None
    elif solver == 'adagradq':
        optimizer = AdagradQ(
            model.parameters(), lr=lr, weight_decay=wd,
            est_bound=hyper['est_bound'], 
            est_step_size=hyper['est_step_size'], 
            est_window=hyper['est_window'],
            mbound=hyper['mbound'],
            polak_ribiere=hyper['polak_ribiere'],
            interp=hyper['interp'] 
        )
        scheduler = None
        
    elif solver == 'pal':
        optimizer = PalOptimizer(
            model.parameters(), writer=None, 
            measuring_step_size=hyper['est_step_size'], 
            max_step_size=hyper['est_bound'],
            direction_adaptation_factor=momentum
        )
        scheduler = None
    elif solver == 'cocob':
        optimizer = COCOB_Backprop(
            model.parameters(), weight_decay=wd
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, 
            nesterov=nesterov, weight_decay=wd
        )
    if lookahead:
        optimizer = Lookahead(optimizer, k=5, alpha=0.5)
        scheduler = None

    if hyper and solver != 'armijo' and solver != 'cgq':
        if 'custom_schedule' in hyper:
            scheduler = CustomLR(optimizer, hyper['custom_schedule'])
        elif 'cosine' in hyper:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, **hyper['cosine']
            )
        else:
            scheduler = None
    else:
        scheduler = None
        
    return optimizer, scheduler, hyper

class CustomLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, schedule, last_epoch=-1):
        if isinstance(schedule, dict):
            self.milestones = list(schedule.keys())
            self.lrs = list(schedule.values())
        else:
            self.scheduler = schedule
        super(CustomLR, self).__init__(optimizer, last_epoch)
    def get_lr(self):
        if hasattr(self,'scheduler'):
            return [
                self.scheduler(base_lr=base_lr, step=self.last_epoch) for base_lr in self.base_lrs
            ]
        else:
            return [
                self.lrs[
                    bisect_right(self.milestones, self.last_epoch)
                ] for _ in self.base_lrs
            ]

class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.hist = []
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.hist.append(val)
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def save_obj(obj, file_name, file_dir='./saved_models'):
    if not os.path.isdir(file_dir): 
        os.makedirs(file_dir)
    
    with open(os.path.join(file_dir,'{}.pkl'.format(file_name)), 'w+b') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(file_name, file_dir='./saved_models'):
    try:
        with open(os.path.join(file_dir,'{}.pkl'.format(file_name)), 'r+b') as f:
            try:
                obj = pickle.load(f)
            except EOFError:
                obj = {}
            return obj
    except FileNotFoundError:
        return {}

def nCorrect(output, target, topk=(1,), regression=False):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        # batch_size = target.size(0)
        if regression:
            pred = output.detach().round()
        else:
            _, pred = output.detach().topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            # res.append(correct_k.mul_(100.0 / batch_size))
            res.append(correct_k.item())
        return res

def draw_line_plot(data, name, save_dir, style_dict, suffix=''):
    """
    Args:
        data - a dict containing data used to draw line plot
        name - file name of figure
        save_dir - directory to save figure
        style_dict - describe style for each line in data: [label, color, line_style, marker]; xlabel, ylabel, title, xticks

    """
    save_dir = save_dir + '/figures/' + suffix
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    lines = []
    matplotlib.rcParams['xtick.major.size'] = 12
    plt.figure(figsize=(8, 8))
    markevery = style_dict['markevery'][0]
    for key in data.keys():
        label = style_dict[key][0].split('_')[0]
        color = style_dict[key][1]
        ls = style_dict[key][2]
        marker = style_dict[key][3]
        lw = style_dict[key][4]
        line, = plt.plot(data[key], color=color, label=label, ls=ls, marker=marker, markevery=markevery, linewidth=lw)
        lines.append(line)
    plt.xlabel(style_dict['xlabel'][0].replace('_', ' '), fontsize=18)
    plt.ylabel(style_dict['ylabel'][0].replace('_', ' '), fontsize=18)
    plt.title(style_dict['title'][0].replace('_', ' '), fontsize=20)
    plt.xticks(style_dict['xticks'][0])
    plt.legend(handles=lines, loc='best', prop={'size': 12})
    plt.grid()
    plt.savefig(save_dir + name)
    plt.close()


def learning_rate_schedule(base_lr, step, cycle_len, mode='step'):
    alpha = (step%cycle_len)/cycle_len # 3*step / cycle_len
    if mode=='cos':
        eta_min = 1/100 # 0.0001
        factor = eta_min+(1-eta_min)*(1+math.cos(math.pi * alpha))/2
        lr = factor*base_lr
    elif mode=='cyc':
        eta_min,eta_max = base_lr/100,base_lr
        if alpha<0.5:
            lr = (1-2*alpha)*eta_max + 2*alpha*eta_min
        else:
            lr = (2-2*alpha)*eta_min + (2*alpha-1)*eta_max
    elif mode=='dnstep':
        if alpha <= 0.5:
            factor = 1.0
        elif alpha < 0.75:
            factor = 0.1
        else:
            factor = 0.01
        lr = factor*base_lr
    elif mode=='imagenet':
        if alpha <= 0.33:
            factor = 1.0
        elif alpha < 0.67:
            factor = 0.1
        else:
            factor = 0.01
        lr = factor*base_lr
    elif mode=='decay':
        factor = 1/(1+step*1e-4)
        lr = factor*base_lr
    else:
        if alpha <= 0.5:
            factor = 1.0
        elif alpha <= 0.9:
            factor = 1.0 - (alpha - 0.5) / 0.4 * 0.99
        else:
            factor = 0.01
        lr = factor*base_lr

    return lr

def get_training_args(
    sgd_epochs=200, startep=0, ds_name='cifar10', batch_size=64, gpu_id='0', model='densenet', nworkers=10,
    solver='sgd', base_lr=0.1, momentum=0.9, nesterov=True, lookahead=False, dp=False, dp_steps=2, dp_start=0, lr_sch=None, scheduler_mode='epoch', 
    est_bound=0.3,est_step_size=0.01,est_decay=0.8,est_window=5,polak_ribiere=False,mbound=0.9, interp='3pt', ls_prob=0.1,
    save=True, save_every=0, suffix='test', save_dir='./saved_models', debug=False, seed=random.randint(0,10000), init_step_size=0.5
):
    def rand_reset(seed=1):
        random.seed(seed)
        torch.manual_seed(random.randint(0,1000))
        torch.cuda.manual_seed_all(random.randint(0,1000))
        np.random.seed(random.randint(0,1000))
    print(gpu_id)
    if str(gpu_id) == 'all':
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    elif str(gpu_id) == 'first_two':
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    elif str(gpu_id) == 'last_two':
        os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
    else:
        if type(gpu_id) is tuple:
            gpu_id = ','.join([str(x) for x in gpu_id])
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    gpu_flag = torch.cuda.is_available()

    print('>>> Random seed: {}'.format(seed))
    rand_reset(seed=seed)

    save_dir += '/'+model+'/'+ds_name+'/'+suffix
    if not os.path.isdir(save_dir): os.makedirs(save_dir)

    frame = inspect.currentframe()
    _,_,_, argvals = inspect.getargvalues(frame)
    print(argvals)

    if ds_name == 'wine':
        train_set, test_set = get_dataset(name=ds_name, target_type=torch.FloatTensor, test_size=0.0)
    else:
        train_set, test_set = get_dataset(name=ds_name, target_type=torch.LongTensor, target_shape=(-1,1), model=model)


    loaders = {}
    loaders['test'] = None if test_set is None else torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, pin_memory=gpu_flag, num_workers=nworkers
    )
    loaders['train'] = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, pin_memory=gpu_flag, num_workers=nworkers
    )

    if ds_name == 'wine':
        num_classes = 3
        num_inputs = 13
        topk=(1,)
        # batch_size=178
        loss_fn = torch.nn.functional.mse_loss
    elif ds_name == 'mnist':
        num_classes = 10
        num_inputs = 784
        topk=(1,)
        # batch_size = 60000
        loss_fn = F.cross_entropy
    elif ds_name == 'imagenet':
        num_classes = 1000
        topk=(1,5)
        loss_fn = F.cross_entropy
    elif ds_name == 'tinyimagenet':
        num_classes = 200
        topk=(1,)
        loss_fn = F.cross_entropy
    else:
        num_classes = 100 if ds_name=='cifar100' else 10
        loss_fn=F.cross_entropy
        topk=(1,)

    if model=='densenet':
        from models.densenet import densenet100
        architecture = densenet100
        arch_args = (num_classes,)
        arch_kwargs = {}
        wd = 3e-4
        lr_mode = 'dnstep'
    elif model=='wrn':
        from models.wide_resnet import wrn_2810
        architecture = wrn_2810
        arch_args = tuple()
        arch_kwargs = {'num_classes':num_classes}
        wd = 5e-4
        lr_mode = 'step'
    elif model in ['resnet','resnet164']:
        from models.resnet import resnet164
        architecture = resnet164
        arch_args = tuple()
        arch_kwargs = {'num_classes':num_classes}
        wd = 3e-4
        lr_mode = 'step'
    elif model=='resnet110':
        from models.resnet import resnet110
        architecture = resnet110
        arch_args = tuple()
        arch_kwargs = {'num_classes':num_classes}
        wd = 3e-4
        lr_mode = 'step'
    elif model=='resnet56':
        from models.resnet import resnet56
        architecture = resnet56
        arch_args = tuple()
        arch_kwargs = {'num_classes':num_classes}
        wd = 3e-4
        lr_mode = 'step'
    elif model=='resnet34':
        from models.resnet_imagenet import resnet34
        architecture = resnet34
        arch_args = tuple()
        arch_kwargs = {'num_classes':num_classes}
        wd = 3e-4
        lr_mode = 'imagenet'
    elif model=='vgg16':
        from models.vgg import vgg16
        architecture = vgg16
        arch_args = tuple()
        arch_kwargs = {'num_classes':num_classes}
        wd = 5e-4 # 5e-4
        lr_mode = 'step'
    elif model=='mlp':
        if ds_name == 'wine':
            from models.mlp import MLPRegressor
            architecture = MLPRegressor
            arch_args = ([13, 1, 1],)
            arch_kwargs = {}
        else:
            from models.mlp import mlp1000
            architecture = mlp1000
            arch_args = (784,10)
            arch_kwargs = {}
        wd = 0.0
        lr_mode = 'decay'
    else:
        raise Exception('Model {} not supported'.format(model))
    

    cycle_len = sgd_epochs if scheduler_mode=='epoch' else 4*len(loaders['train'])
    hyper_init = {
        'lr':base_lr, 'wd': wd, 'momentum':momentum, 'nesterov': nesterov,
        'n_batches_per_epoch':len(loaders['train']), # for armijo optimizer
        'init_step_size': init_step_size, # for armijo optimizer
        'est_step_size':est_step_size,'est_bound':est_bound,'ls_prob':ls_prob,# for CGQ
        'est_window':est_window,'mbound':mbound,'polak_ribiere':polak_ribiere,'interp':interp,# for CGQ
    }
    if lr_sch=='cosine':
        hyper_init['cosine'] = {'T_max': sgd_epochs, 'eta_min': 1e-2*base_lr}
    elif lr_sch is not None:
        hyper_init['custom_schedule'] = partial(learning_rate_schedule, cycle_len=cycle_len, mode=lr_mode)

    ### initial training
    print('Begin Training...')
    init_suffix = 'inittrain'

    args_train = {
        'startep': startep,
        'sgdepochs': sgd_epochs, 
        'batch_size': batch_size, 
        'suffix': init_suffix, 
        'save': save,
        'save_every': save_every,
        'test_along': True,
        'loss_fn': loss_fn,
        'result_path':save_dir,
        'hyper': hyper_init, 
        'topk':topk,
        'solver': solver, 
        'lookahead':lookahead, 
        'dp':dp, 'dp_steps': dp_steps, 'dp_start': dp_start,
        'scheduler_mode':scheduler_mode,
        'est_decay':est_decay,
        'debug': debug
    }
    return architecture, arch_args, arch_kwargs, loaders, args_train



if __name__ == '__main__':
    inputs = torch.arange(288).view(2,1,12, 12)
    kernel = torch.arange(36).view(4, 1, 3, 3)
    feature_map = _conv2d(inputs, kernel)
    print(inputs.shape, kernel.shape, feature_map.shape)
    
