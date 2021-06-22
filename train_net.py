import os,fire,time,math,torch,random,inspect,contextlib
import numpy as np
import torch.nn.functional as F
from functools import partial

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from helpers import get_dataset, get_optimizer, save_obj, load_obj, nCorrect, get_training_args, draw_line_plot
from collections import defaultdict


def rand_reset(seed=1):
    random.seed(seed)
    torch.manual_seed(random.randint(0,1000))
    torch.cuda.manual_seed_all(random.randint(0,1000))
    np.random.seed(random.randint(0,1000))

def test_epoch(model, loader, topk=(1,), loss_fn=F.cross_entropy, mdl_kwargs={}):
    # Model on eval mode
    model.eval()
    with torch.no_grad():
        loss_sum, num_samples = 0.0, 0
        running_corrects = [0.0 for _ in topk]
        for _iter, (inputs, target) in enumerate(loader):
            # Create vaiables
            if torch.cuda.is_available():
                inputs, target = inputs.cuda(non_blocking=True), target.cuda(non_blocking=True)

            # compute output
            output = model(inputs, **mdl_kwargs)

            if loss_fn.__name__ == 'cross_entropy':
                loss = loss_fn(output, target)
            else:
                loss = loss_fn(output, target.view_as(output))

            loss_sum += loss * len(target)

            for i, topx in enumerate(nCorrect(output.detach().data, target.data, topk=topk, regression=loss_fn.__name__!='cross_entropy')):
                running_corrects[i] += topx
            # measure accuracy
            num_samples += len(target)
            
        res = {
            'acc{}'.format(x): running_corrects[i]*100.0/num_samples for i,x in enumerate(topk)
        }
        res['loss'] = loss_sum.item() / num_samples

    model.train()
    return res

def convert_time(elapse):
    if elapse<120:
        res = '{:8.3f}s'.format(elapse)
    elif elapse<1800:
        res = '{:6.3f}min'.format(elapse/60)
    else:
        res = '{:7.3f}hr'.format(elapse/3600)
    return res

def fix_model_for_line_search(model, fix=True):
    # was_training = model.training()
    for m in model.modules():
        if 'BatchNorm' in type(m).__name__:
            m.track_running_stats = not fix # True if use batch_stats; False if accumulative_stats
        if 'Dropout' in type(m).__name__:
            m.fix_mask = fix
    # return was_training


@contextlib.contextmanager
def random_seed_torch(seed):
    """
    source: https://github.com/IssamLaradji/sls/
    """
    cpu_rng_state = torch.get_rng_state()
    gpu_rng_state = torch.cuda.get_rng_state(0)

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    try:
        yield
    finally:
        torch.set_rng_state(cpu_rng_state)
        torch.cuda.set_rng_state(gpu_rng_state)

        
def train(
    architecture, mdl_args, mdl_kwargs, loaders, hyper={}, init_pt=None, t0_pt=None,
    sgdepochs=100, startep=0, batch_size=64, topk=(1,), test_along=True, ls_prob=0.1, save_every=0,
    solver='sgd',lookahead=False, dp=False, dp_steps=2, dp_start=0, loss_fn=F.cross_entropy, scheduler_mode='epoch',
    est_decay=0.8, save=False, suffix= '', result_path = './saved_models',debug=False, seed=random.randint(0,10000), 
):
    """
    function that fires the training process
    Args:
        architecture (class/function) - python class for/function that returns a given architecture
        mdl_args (tuple) - positional arguments to instanciate a model object
        mdl_kwargs (dict) - keyword arguments to instanciate a model object
        loaders (dict) - dictionary of training and testing data loaders
        hyper (dict) - hyperparameters for local solver (default {})
        init_pt (state_dict) - state_dict of initial point to be loaded to model before training (default None)
        t0_pt (state_dict) - used when computing parameter distance from during training (default None)
        sgdepochs (int) - number of training epochs (default 100)
        startep (int) - starting epoch number (default 0)
        batch_size (int) - size of minibatch (default 64)
        topk (tuple) - used when computing (top-k) prediction error (default (1,))
        test_along (bool) - evaluate test set every epoch? (default True)
        solver (str) - specify the local solver (sgd/adagrad/adam/armijo/cgq/cocob) (default sgd)
        lookahead (bool) - use lookahead optimizer on top of other optimizers? (default False)
        loss_fn (function) - the loss function used for training (default F.cross_entropy)
        scheduler_mode (str) - call lr scheduler in each batch or each epoch (epoch/iter) (default epoch)
        est_decay (float) - sample distance decaying factor(for CGQ solver) (default 0.8)
        save (bool) - save trained models? (default True)
        suffix (str) - a suffix used as a unique identifier when saving models (default '')
        result_path (str) - path to directory where data should be loaded from/downloaded
            (default './saved_models')
        debug (bool) - run in debug mode? (default False)
        seed (int) - manually set the random seed (default random)
    """
    # mse_loss: FloatTensor cross_entropy: LongTensor
    gpu_flag = torch.cuda.is_available()
    model = architecture(*mdl_args,**mdl_kwargs)
    if gpu_flag:
        if torch.cuda.device_count() > 1:
            print('miltiple gpu training!')
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.cuda()
    
    numparam = 0
    for name, param in model.named_parameters():
        numparam += param.numel()
    print('#Parameters:',numparam)

    if init_pt is not None:
        model.load_state_dict(init_pt)
        
    checksum = 0.0
    for k,v in model.named_parameters():
        checksum += v.sum().item()
    print('param checksum:',checksum)

    if t0_pt is not None:
        orig_norm = 0.
        orig_dir = {}
        for k,v in t0_pt.items():
            if 'weight' in k or 'bias' in k:
                orig_dir[k] = v.clone()
                try:
                    orig_norm += v.norm()**2
                except RuntimeError:
                    print(k,v.shape,v)
                    raise Exception('ffff')
        orig_norm = torch.sqrt(orig_norm)
    

    fig_path = os.path.join(result_path, 'figures')
    if not os.path.isdir(fig_path): os.makedirs(fig_path)

    optimizer,scheduler,_ = get_optimizer(model, hyper, sgdepochs, solver=solver, lookahead=lookahead, dp=dp, dp_steps=dp_steps)

    start_time = time.time()
    train_hist_loss,train_hist_acc1,train_hist_acc5,test_hist_loss,test_hist_acc1,test_hist_acc5,temp_lr_hist,temp_momentum_hist = [],[],[],[],[],[],[],[]

    est_step_size = hyper.get('est_step_size',0.0)
    est_bound = hyper.get('est_bound',0.0)
    epoch_time = []

    if startep>0:
        model_name = '{}_ep{}.dat'.format(suffix,startep-1)
        file_dir = os.path.join(result_path, model_name)
        if os.path.isfile(file_dir):
            checkpoint = torch.load(file_dir)
            model.load_state_dict(checkpoint)
            print('epoch{} model loaded.'.format(startep-1))
    model.train()
    print(f'train_iter:{len(loaders["train"])}, test_iter:{len(loaders["test"])}')
    cgq_mode = True
    for epoch in range(startep, sgdepochs):
        _start = time.time()

        train_loss, num_samples = 0.0, 0
        running_corrects = [0.0 for _ in topk]

        temp_lr = 0.
        temp_momentum = 0.
        for _iter, (inputs, target) in enumerate(loaders['train']):
            if gpu_flag:
                inputs, target = inputs.cuda(non_blocking=True), target.cuda(non_blocking=True)

            # output = model(inputs)
            def closure(seed=int(time.time()), return_out=False):
                with random_seed_torch(seed):
                    output = model(inputs)
                    # assert loss_fn.__name__ == 'cross_entropy':
                    _loss = loss_fn(output, target)
                if return_out:
                    return _loss, output
                else:
                    return _loss
            if solver=='armijo': # Stochastic Armijo Line Search
                optimizer.zero_grad()
                loss,output = optimizer.step(closure=closure)
                # with torch.no_grad():
                #     loss = closure()
                temp_lr += optimizer.state.get('lr',0.0)
            elif solver in ['cgq','adamq', 'adagradq']: # Conjugate Gradient with Quadratic Line Search
                ### optimizer.zero_grad() and loss.backward() are in optimizer.step()
                determ_closure = partial(closure,seed=int(time.time())) # deterministic closure
                loss, output = optimizer.step(determ_closure)

                temp_lr += optimizer.param_groups[0].get('lr',0.0)
                temp_momentum += optimizer.param_groups[0].get('momentum',0.0)
            elif solver == 'dp':
                start_counting = epoch >= dp_start
                loss = optimizer.step(closure=closure, start_counting=start_counting)
                temp_lr += optimizer.optimizer.param_groups[0].get('lr',0.0)
                temp_momentum += optimizer.optimizer.param_groups[0].get('momentum',0.0)
            elif solver == 'pal':
                optimizer.zero_grad()
                def closure_pal(backward=True):
                    out_ = model(inputs)
                    loss_ = loss_fn(out_, target)
                    if backward:
                        loss_.backward()
                    return loss_, out_

                loss, output, _ = optimizer.step(closure_pal)
                temp_lr += optimizer.param_groups[0].get('lr',0.0)
                temp_momentum += optimizer.param_groups[0].get('direction_adaptation_factor',0.0)

            else: # other optimizer (SGD, Adam, lookahead, etc...)
                optimizer.zero_grad() 
                loss,output = closure(return_out=True)
                loss.backward()
                optimizer.step()
                if solver != 'cocob':
                    temp_lr += optimizer.param_groups[0].get('lr',0.0)
                    temp_momentum += optimizer.param_groups[0].get('momentum',0.0)

            train_loss += loss.detach() * inputs.size(0)
            for i,topx in enumerate(nCorrect(output.detach().data, target.data, topk=topk, regression=loss_fn.__name__!='cross_entropy')):
                running_corrects[i] += topx

            num_samples += len(target)

            if scheduler is not None:
                if scheduler_mode=='iter' or (scheduler_mode=='epoch' and _iter==0):
                    scheduler.step()
                    
        _end = time.time()
        _elapse = convert_time(_end-_start)
        epoch_time.append(_end-_start)

        temp_lr /= _iter+1
        temp_momentum /= _iter+1
        temp_lr_hist.append(temp_lr)
        temp_momentum_hist.append(temp_momentum)
        tr_res = {
            'acc{}'.format(x): running_corrects[i]*100.0/num_samples for i,x in enumerate(topk)
        }
        tr_res['loss'] = train_loss.item() / num_samples

        train_hist_loss.append(tr_res['loss'])
        train_hist_acc1.append(tr_res['acc1'])
        train_hist_acc5.append(tr_res['acc5'] if 'acc5' in tr_res.keys() else None)


        # ### switch to sgd
        # if solver in ['cgq', 'adamq','adagradq'] and est_decay<1:
        #     if cgq_mode and temp_lr<hyper['lr']*0.1:
        #     # if cgq_mode and epoch>=sgdepochs/2:
        #         cgq_mode = False
        #         cyc_len = sgdepochs-epoch
        #         thres_lr = temp_lr
        #         eta_min = 0.001*hyper['lr']/temp_lr # 0.01 eta_min=0.001*hyper['lr']/temp_lr
        #         optimizer.ls_prob = 0
        #         # optimizer.param_groups[0]['momentum'] = 0.9
        #     if not cgq_mode:
        #         alpha = ((epoch+cyc_len-sgdepochs)%cyc_len)/cyc_len
        #         factor = eta_min+(1-eta_min)*(1+math.cos(math.pi * alpha))/2
        #         optimizer.param_groups[0]['lr'] = factor*thres_lr # temp_lr * 0.9 # 1e-3


        ### decay est_step_size
        if solver in ['cgq', 'adamq','adagradq'] and est_decay<1:
            if temp_lr<est_step_size:
                est_step_size *= est_decay
            if lookahead:
                optimizer.optimizer.reset_estimate(est_step_size=est_step_size)
            else:
                optimizer.reset_estimate(est_step_size=est_step_size)

        # ### cosine anneal estbound
        # if solver in ['cgq', 'adamq','adagradq'] and est_decay<1:
        #     alpha = (epoch%sgdepochs)/sgdepochs
        #     factor = 0.01+(1-0.01)*(1+math.cos(math.pi * alpha))/2
        #     if lookahead:
        #         optimizer.optimizer.reset_estimate(est_bound=est_bound*factor)
        #     else:
        #         optimizer.reset_estimate(est_bound=est_bound*factor)


        temp_norm = 0.
        if t0_pt is not None:
            for k,v in model.named_parameters():
                if k in orig_dir:
                    temp_norm += (v.data-orig_dir[k]).norm()**2
            temp_norm = torch.sqrt(temp_norm).item()
        # train_loss, train_error = trainer.eval()
        if test_along and loaders['test'] is not None:
            ts_res = test_epoch(
                model=model, loader=loaders['test'], loss_fn=loss_fn, topk=topk,
            )
            tr_acc = ', '.join(
                ( '{}={:6.2f}%'.format('acc'+str(x), tr_res['acc'+str(x)]) for x in topk)
            )
            ts_acc = ', '.join(
                ( '{}={:6.2f}%'.format('acc'+str(x), ts_res['acc'+str(x)]) for x in topk)
            )

            values = (
                epoch, tr_res['loss'], tr_acc, ts_res['loss'], ts_acc, 
                temp_lr, temp_momentum, est_step_size, _elapse
            )
            print('[Epoch{:3}] Train:(loss={:8.4f}, {}); Test:(loss={:8.4f}, {}); lr={:.2e}; m={:.2e}; est_step={:.2e}, Time={}'.format(*values))

            test_hist_loss.append(ts_res['loss'])
            test_hist_acc1.append(ts_res['acc1'])
            test_hist_acc5.append(ts_res['acc5'] if 'acc5' in ts_res.keys() else None)
        else:
            values = (epoch, tr_res['loss'], tr_acc, temp_lr, _elapse)
            print('[Epoch{:3}] Train:(loss={:8.4f}, {}); lr={:.2e}; Time={}'.format(*values))

        if save and save_every>0 and (epoch+1)%save_every==0:
            if not os.path.isdir(result_path): os.makedirs(result_path)
            model_name = '{}_ep{}.dat'.format(suffix,epoch)
            opt_name = '{}_ep{}_opt.dat'.format(suffix,epoch)
            torch.save(model.state_dict(), os.path.join(result_path, model_name))
            torch.save(optimizer.state_dict(), os.path.join(result_path, opt_name))
            print(f'epoch{epoch} model saved.')

    
    if loaders['test'] is not None:
        ts_res = test_epoch(
            model=model, loader=loaders['test'], loss_fn=loss_fn, topk=topk,
        )
        ts_acc = ', '.join(
            ( '{}={:6.2f}%'.format('acc'+str(x), ts_res['acc'+str(x)]) for x in topk)
        )
        print('>>> Testing loss: {:10.4f}'.format(ts_res['loss']))
        print('>>> Testing acc: {}'.format(ts_acc))
    else:
        ts_res = {'loss':0.,'acc1':0.,'corr':0.}
        ts_acc = 'acc1=0.0'

    end_time = time.time()
    elapse = convert_time(end_time-start_time)
    print(f'>>> Total training time: {elapse}, avg epoch time: {convert_time(np.mean(epoch_time))}')

    # if len(train_hist)>0 or len(test_hist)>0:
    fig_label = 'sgd{}_{}'.format(sgdepochs, suffix)
    title = ''
    if ts_res['acc1']>0:
        title += '{:.3f}'.format(ts_res['acc1'])
    plt.figure()
    plt.plot(train_hist_acc1, 'r', ls='-', label='training accuracy(%)')
    plt.plot(test_hist_acc1, 'b', ls='--', label='testing accuracy(%)')
    plt.legend(loc='best')
    plt.xlabel('steps')
    plt.title('Training and Testing Error (%={})'.format(title))
    plt.grid()
    plt.savefig(os.path.join(fig_path, fig_label+'_err.pdf'), format='pdf')
    plt.close()

    title2 = ''
    if ts_res['acc1']>0:
        title2 += '{:.4f}'.format(ts_res['acc1'])
    plt.figure()
    plt.plot(train_hist_loss, 'r', ls='-', label='training loss')
    plt.plot(test_hist_loss, 'b', ls='--', label='testing loss')
    plt.legend(loc='best')
    plt.xlabel('steps')
    plt.title('Training and Testing loss ({})'.format(title2))
    plt.grid()
    plt.savefig(os.path.join(fig_path, fig_label+'_loss.pdf'), format='pdf')
    plt.close()
    
    if save:
        if not os.path.isdir(result_path): os.makedirs(result_path)
        model_name = '{}.dat'.format(suffix)
        torch.save(model.state_dict(), os.path.join(result_path, model_name))
        opt_name = '{}_opt.dat'.format(suffix)
        torch.save(optimizer.state_dict(), os.path.join(result_path, opt_name))
        print('trained model saved.')
    

    # if debug:
    #     for k,v in model.named_parameters():
    #         print(k,v.data.cpu().numpy().flatten())
    data_dict_for_plot = {'train_loss_hist': train_hist_loss}
    data_dict_for_plot['train_acc1_hist'] = train_hist_acc1
    data_dict_for_plot['train_acc5_hist'] = train_hist_acc5
    data_dict_for_plot['test_loss_hist'] = test_hist_loss
    data_dict_for_plot['test_acc1_hist'] = test_hist_acc1
    data_dict_for_plot['test_acc5_hist'] = test_hist_acc5
    data_dict_for_plot['temp_lr_hist'] = temp_lr_hist
    data_dict_for_plot['temp_momentum_hist'] = temp_momentum_hist
    return model, data_dict_for_plot

def comparing_experiment(
    sgd_epochs=200, startep=0, ds_name='cifar10', batch_size=64, gpu_id='0', nworkers=10,
    model='densenet', scheduler_mode='epoch', save_dir='./saved_models', 
    save=True, debug=False, seed=random.randint(0,10000), figure_suffix='test/'
):
    """
    Train the same architecture on the same dataset with different optimizers
    Args:
        sgd_epochs (int) - number of training epochs (default 200)
        startep (int) - starting epoch number (default 0)
        ds_name (str) - dataset name used for training(wine/mnist/cifar10/cifar100/imagenet/tinyimagenet/...) (default cifar10)
        batch_size (int) - size of minibatch (default 64)
        gpu_id (str) - GPU number to be used (default 0)
        model (str) - specify the architecture (densenet/resnet/resnet110/resnet34/wrn/vgg16/mlp) (default densenet)
        plot_data (list) - data to plot (a subset of [train_hist_loss, train_hist_err, test_hist_loss, test_hist_err]) (default [train_hist_loss])
        scheduler_mode (str) - call lr scheduler in each batch or each epoch (epoch/iter) (default epoch)
        save (bool) - save trained models? (default True)
        save_dir (str) - path to directory where data should be loaded from/downloaded
            (default './saved_models')
        debug (bool) - run in debug mode? (default False)
        seed (int) - manually set the random seed (default random)
    """
    # parameters that is not related to optimizers
    args = {
        'sgd_epochs': sgd_epochs, 
        'startep': startep, 
        'ds_name': ds_name, 
        'batch_size': batch_size, 
        'gpu_id': gpu_id, 
        'nworkers': nworkers,
        'model': model,  
        'scheduler_mode': scheduler_mode, 
        'save': save,
        'save_every': 1 if ds_name=='imagenet' else 50,
        'save_dir': save_dir, 
        'debug': debug, 
        'seed': seed
    }
    # benchmark solvers and target plotting data are hard-coded 

    # 'sgd','adam', 'cocob', 'lookahead', 'armijo', 'sgd_lrschcosine',
    # 'pal_mom0.4_estbound3.16_eststep0.5', # PAL algorithm
    # 'cgq3_momPR', 'cgq3_momFR', 'cgq3_momHS', 'cgq3_momDY',
    # 'cgq3_momPR_mbound0.8_nest_estbound0.1_lsprob0.1',
    # 'cgq2_momPR_mbound0.8_nest_estbound0.1_lsprob0.1',
    if model=='vgg16':
        solvers = [
            'cgq2_momPR_mbound0.8_estbound0.05_lsprob0.1',
            'cgq3_momPR_mbound0.8_estbound0.05_lsprob0.1_window3',
        ]
    elif model=='densenet':
        solvers = [
            'cgq2_momPR_mbound0.8_nest_estbound0.1_lsprob0.1',
            'cgq3_momPR_mbound0.8_nest_estbound0.1_lsprob0.1',
        ]
    else:
        solvers = [
            'cgq2_momPR_mbound0.8_estbound0.3_lsprob0.1',
            'cgq3_momPR_mbound0.8_estbound0.3_lsprob0.1',
        ]

    # alphas = [0.01,0.05,0.1,0.3,0.5,0.7,0.9,1.1]
    # betas = [0.0,0.2,0.4,0.6,0.8,1.0,1.2]
    # args['save'] = False
    # solvers = [
    #     f'cgq3_nest_momPR_mbound{b}_estbound{a}_lsprob0.1' for a in alphas for b in betas
    # ]

    plot_data=['train_loss_hist', 'train_acc1_hist', 'train_acc5_hist', 'test_loss_hist', 'test_acc1_hist', 'test_acc5_hist', 'temp_lr_hist', 'temp_momentum_hist']

    data = []
    for target in plot_data:
        data.append(defaultdict(list))
    style_dict = defaultdict(list)
    color_list = ['red', 'blue', 'gold', 'green', 'orange', 'purple', 'black', 'cyan', 'magenta', 'olive', 'hotpink']
    marker_list = ['o', 'v', 's', '*', 'p', 'P', '1', 'X', 'd', 'H']
    color_pt = 0
    marker_pt = 0
    for solver_name in solvers:
        args['suffix'] = solver_name
        args['lookahead'] = False
        if solver_name.startswith('sgd'):
            # parameters for sgd with fixed learning rate and fixed nesterov momentum
            args['solver'] = 'sgd'
            if model == 'vgg16':
                args['base_lr'] = 0.02
            elif ds_name == 'mnist':
                args['base_lr'] = 0.01
            else:
                args['base_lr'] = 0.1
            args['momentum'] = 0.9
            args['nesterov'] = True
            args['lr_sch'] = None
            for name in solver_name.split('_'):
                if name.startswith('lrsch'):
                    args['lr_sch'] = name[5:]
            style = [solver_name, color_list[color_pt], '--', marker_list[marker_pt], 1]
            style_dict[solver_name] = style
            color_pt += 1
            marker_pt += 1
        elif solver_name == 'adam':
            # parameters for adam
            args['solver'] = 'adam'
            if model == 'vgg16':
                args['base_lr'] = 0.0001
            else:
                args['base_lr'] = 0.001
            args['lr_sch'] = None
            style = [solver_name, color_list[color_pt], '--', marker_list[marker_pt], 1]
            style_dict[solver_name] = style
            color_pt += 1
            marker_pt += 1
        elif solver_name == 'adagrad':
            # parameters for adam
            args['solver'] = 'adagrad'
            if model == 'vgg16':
                args['base_lr'] = 0.01
            else:
                args['base_lr'] = 0.01
            args['lr_sch'] = None
            style = [solver_name, color_list[color_pt], '--', marker_list[marker_pt], 1]
            style_dict[solver_name] = style
            color_pt += 1
            marker_pt += 1
        elif solver_name.startswith('cocob'):
            args['solver'] = 'cocob'
            args['lr_sch'] = None
            style = [solver_name, color_list[color_pt], '--', marker_list[marker_pt], 1]
            style_dict[solver_name] = style
            color_pt += 1
            marker_pt += 1
        elif solver_name.startswith('lookahead'):
            # parameters for Lookahead + SGD with fixed learning rate and fixed nesterov momentum
            args['solver'] = 'lookahead'
            args['lookahead'] = True
            if model == 'vgg16':
                args['base_lr'] = 0.02 # if ds_name=='svhn' else 0.05
            elif ds_name == 'mnist':
                args['base_lr'] = 0.01
            else:
                args['base_lr'] = 0.1
            args['momentum'] = 0.9
            args['nesterov'] = True
            args['lr_sch'] = None
            style = [solver_name, color_list[color_pt], '--', marker_list[marker_pt], 1]
            style_dict[solver_name] = style
            color_pt += 1
            marker_pt += 1
        elif solver_name.startswith('armijo'):
            # parameters for SGD + Armijo rule
            args['solver'] = 'armijo'
            args['init_step_size'] = 0.5
            args['lr_sch'] = None
            style = [solver_name, color_list[color_pt], '--', marker_list[marker_pt], 1]
            style_dict[solver_name] = style
            color_pt += 1
            marker_pt += 1
        elif solver_name.startswith('cgq'):
            args['solver'] = 'cgq'
            args['base_lr'] = 0.1
            args['momentum'] = 0.9
            args['nesterov'] = False
            args['est_bound'] = 0.3 # 0.3
            args['est_step_size'] = 0.1
            args['est_window'] = 3
            args['polak_ribiere'] = None
            args['mbound'] = 0.8
            args['ls_prob'] = 0.1 # line search probability
            args['est_decay'] = 0.8 # 0.8
            args['interp'] = '2pt'
            args['lr_sch'] = None
            for name in solver_name.split('_'):
                if name.startswith('mom'):
                    try:
                        args['momentum'] = float(name[3:])
                    except ValueError:
                        args['polak_ribiere'] = str(name[3:])
                elif name.startswith('nest'):
                    args['nesterov'] = True
                elif name.startswith('eststep'):
                    try:
                        args['est_step_size'] = float(name[7:])
                    except ValueError:
                        raise Exception(solver_name)
                elif name.startswith('mbound'):
                    try:
                        args['mbound'] = float(name[6:])
                        args['momentum'] = min(args['mbound'],args['momentum'])
                        args['nesterov'] = args['momentum']>0
                    except ValueError:
                        print('given invalid mbound!')
                elif name.startswith('estbound'):
                    try:
                        args['est_bound'] = float(name[8:])
                        args['base_lr'] = min(args['est_bound'],args['base_lr'])
                        args['est_step_size'] = min(args['est_bound']/3,args['est_step_size'])
                    except ValueError:
                        print('given invalid mbound!')
                elif name.startswith('lsprob'):
                    try:
                        args['ls_prob'] = float(name[6:])
                    except ValueError:
                        print('given invalid line search probability!')
                elif name.startswith('window'):
                    try:
                        args['est_window'] = int(name[6:])
                    except ValueError:
                        print('given invalid estimation window size')
                elif name.startswith('interp'):
                    try:
                        args['interp'] = str(name[6:])
                    except ValueError:
                        print('given invalid interpolation method!')
                elif name.startswith('lookahead'):
                    args['lookahead'] = True
            legend_name = 'cgq(2points)'
            if solver_name.startswith('cgq3'):
                legend_name = 'cgq(LS)'
                args['interp'] = '3pt'
            style = [legend_name, color_list[color_pt], '-', marker_list[marker_pt], 2]
            style_dict[solver_name] = style
            color_pt += 1
            marker_pt += 1

        elif solver_name.startswith('pal'):
            args['solver'] = 'pal'
            args['momentum'] = 0.4
            args['est_bound'] = 3.16
            args['est_step_size'] = 0.5
            args['lr_sch'] = None
            for name in solver_name.split('_'):
                if name.startswith('mom'):
                    try:
                        args['momentum'] = float(name[3:])
                    except ValueError:
                        raise Exception(solver_name)
                elif name.startswith('eststep'):
                    try:
                        args['est_step_size'] = float(name[7:])
                    except ValueError:
                        raise Exception(solver_name)
                elif name.startswith('estbound'):
                    try:
                        args['est_bound'] = float(name[8:])
                    except ValueError:
                        print('given invalid mbound!')
                elif name.startswith('lookahead'):
                    args['lookahead'] = True
            legend_name = 'PAL'
            style = [legend_name, color_list[color_pt], '--', marker_list[marker_pt], 1]
            style_dict[solver_name] = style
            color_pt += 1
            marker_pt += 1

        print("training for {}".format(solver_name))
        architecture, arch_args, arch_kwargs, loaders, args_train = get_training_args(**args)
        mdl, data_dict_for_plot = train(architecture, arch_args, arch_kwargs, loaders, **args_train)
        for i, target in enumerate(plot_data):
            data[i][solver_name] = data_dict_for_plot[target]
            step = np.maximum(int(sgd_epochs / 5), 1)
            style_dict['xticks'] = [np.arange(0, sgd_epochs+1, step)]
            style_dict['markevery'] = [int( sgd_epochs / 5)]
        
    # Plotting
    for i, target in enumerate(plot_data):
        style_dict['xlabel'] = ['epoch']
        y_label = target.replace('_hist', '')
        style_dict['ylabel'] = [y_label]
        title = ds_name + '_' + model + '_' + y_label
        style_dict['title'] = [title]
        name = title + '.pdf'
        draw_line_plot(data[i], name, save_dir, style_dict, figure_suffix)


if __name__ == '__main__':
    # Usage example:
    # python train_net.py --sgd_epochs=200 --ds_name=cifar10 --batch_size=128 --gpu_id=0 --model=resnet --figure_suffix='test/' --seed=1234
    fire.Fire(comparing_experiment)
    
