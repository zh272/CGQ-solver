import os,torch,time,copy,math,bisect,random,operator,contextlib
import numpy as np
from collections import defaultdict
from torch.optim.optimizer import Optimizer, required
from typing import List
from torch import Tensor

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



def estimate_min_np(out_val, in_val, bound=None, slope=None, dim=2, plot=False, suffix=''):
    x_mat = np.concatenate([
        in_val.reshape(-1,1)**2,
        in_val.reshape(-1,1),
        np.ones((len(in_val),1), dtype=float)
    ],axis=1)
    if slope is not None:
        x_mat = np.append(x_mat, [[2.0*in_val[0], 1.0, 0.0]], axis=0)
        out_val = np.append(out_val, [slope], axis=0)
    try:
        params = np.matmul(np.matmul(np.linalg.pinv(np.matmul(x_mat.T,x_mat)),x_mat.T),out_val)
    except np.linalg.LinAlgError:
        return None, False, None, (None,None,None)
    a,b,c = iter(params.flatten())

    pos_def = True if a>0 else False

    if pos_def:
        xmin = -b/(2*a)
    else:
        min_idx = np.argmin(out_val[:len(in_val)])
        xmin = in_val[min_idx]

    if bound is not None:
        xmin = min(xmin, bound)
    lmin = a*xmin**2+b*xmin+c

    # # Plot the surface.
    if plot and lmin is not None:
        window = (max(np.max(in_val),xmin) - min(np.min(in_val),xmin))*1.2
        mid = (max(np.max(in_val),xmin) + min(np.min(in_val),xmin))/2
        X = np.arange(mid-window/2, mid+window/2, window/20)
        Z = a*X**2+b*X+c
    
        fig, ax = plt.subplots()
        surf = ax.plot(X,Z)
        
        for ii in range(len(in_val)):
            marker = 'x' if ii==0 else 'o'
            ax.scatter(in_val[ii],out_val[ii], marker=marker, c='k',s=20*(1.2-ii/len(out_val)))
        ax.scatter(xmin,lmin, marker='*', c='k',s=50)
        plt.title('Estimated min: {:.4e}'.format(lmin))
        plt.grid()
        # plt.show()
        plt.savefig('./estimate_surf_{}.pdf'.format(suffix), format='pdf')
        plt.close()
        
    return lmin, pos_def, xmin, (a,b,c)

def move_along_grad(param_group, step_size):
    for param in param_group['params']:
        if param.grad is None:
            continue
        param.add_(param.grad, alpha=step_size)

@torch.no_grad()
def opt_lr(param_group, closure, init_loss, slope, grad_norm, method='2pt', plot=False,debug=False, suffix=None,
            est_window=5, est_bound=0.1, est_step_size=1e-2):
    ### initial point
    temp_losses = [init_loss.item()]
    temp_losses_all = [init_loss.item()]
    relative_loc = 0.
    in_val=[relative_loc]
    in_val_all = [relative_loc]
    # est_step_size /= grad_norm

    ### move one step
    for _ in range(1):# if method=='2pt' else 2):
        move_along_grad(param_group, -est_step_size)
        relative_loc += 1.
        loss = closure()
        temp_losses.append(loss.item())
        in_val.append(relative_loc)
        temp_losses_all.append(loss.item())
        in_val_all.append(relative_loc)

    a_list, b_list, c_list = [],[],[]
    for k in range(est_window):
        # prev_xmin = xmin if k>0 else None
        lmin, pos_def,xmin, (a,b,c) = estimate_min_np(np.array(temp_losses), np.array(in_val), 
                                bound=est_bound/est_step_size,slope=slope*est_step_size, dim=1,plot=plot,
                                suffix=suffix+f'_{k}' if suffix is not None else None)
        a_list.append(a)
        b_list.append(b)
        c_list.append(c)

        ### evaluate estimated optimal
        est_lr = xmin*est_step_size
        move_along_grad(param_group, relative_loc*est_step_size-est_lr)
        relative_loc = xmin

        if est_window>1: # if est_window<=1, do not perform feedback
            real_opt_loss = closure().item()
            if np.isnan(real_opt_loss):
                print('opt_lr', est_lr, relative_loc, est_step_size)
            temp_losses_all.append(real_opt_loss)
            in_val_all.append(xmin)

            if method == '2pt':
                temp_losses[1] = real_opt_loss
                in_val[1] = xmin
            else:
                temp_losses.append(real_opt_loss)
                in_val.append(xmin)

            if real_opt_loss<init_loss.item(): # (abs(lmin-real_opt_loss)/lmin<1e-1 # or (prev_xmin is not None and abs(xmin-prev_xmin)/xmin<1e-1)
                break
            
        ### check stop criteria
        if not pos_def:
            break

    ### plot bad estimation
    if plot:
        if suffix is None:
            suffix = 'debug'
        curves,legends = [],[]
        ### revert parameters
        fig, ax = plt.subplots()
        move_along_grad(param_group, relative_loc*est_step_size) # move to origin
        for idx, (a,b,c) in enumerate(zip(a_list,b_list,c_list)):
            _min = min(np.min(in_val),xmin)
            _max = max(np.max(in_val),xmin)
            window = (_max - _min)*1.2
            mid = (_max + _min)/2
            X = np.arange(mid-window/2, mid+window/2, window/20)
            Z = a*X**2+b*X+c
            if idx==0:
                curves.append(ax.plot(X*est_step_size,Z, c='r',alpha=0.7)[0])
                legends.append('2-point method')
            else:
                curves.append(ax.plot(X*est_step_size,Z, c='b',alpha=0.7)[0])
                legends.append('LS method')

        real_surface_y = [init_loss.item()]
        real_surface_x = [0.,]
        _iter = 100
        _step = 1.2*relative_loc*est_step_size/_iter
        for kkk in range(_iter):
            ### update parameters
            move_along_grad(param_group, -_step)
            loss = closure()
            real_surface_y.append(loss.item())
            real_surface_x.append(kkk+1)
            
        ### revert parameters
        move_along_grad(param_group, _step*_iter-relative_loc*est_step_size)
            
        curves.append(ax.plot([x*_step for x in real_surface_x],real_surface_y, c='k',alpha=0.7)[0])
        legends.append('Loss Landscape')
        for ii in range(len(in_val_all)):
            marker = 'x' if ii==0 else 'o'
            ax.scatter(in_val_all[ii]*est_step_size,temp_losses_all[ii], marker=marker, c='k',s=20*(1.2-ii/len(in_val_all)))
        if xmin is not None:
            ax.scatter(xmin*est_step_size,lmin, marker='*', c='k',s=50)
            print('opt_lr',xmin*est_step_size,grad_norm)

        plt.xlabel('stepsize')
        plt.title(f'Line_Search_{suffix}{"slope{:.2e}".format(slope) if slope is not None else ""}')
        plt.legend(curves, legends, loc='best')
        plt.savefig('./debug/mnist/{}_lr{:.2e}_{}.pdf'.format(suffix, est_lr,k), format='pdf')
        plt.close()
    return est_lr


class CGQ(Optimizer):
    r"""Implements Conjugate Gradient with Quadratic Line Search.
    """
    def __init__(
        self, params, lr=required, momentum=0, dampening=0, weight_decay=0, nesterov=False, ls_prob=0.1,
        est_bound=0.5, est_step_size=0.02, est_window=5, mbound=0.9, polak_ribiere=None, interp='3pt'
    ):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,polak_ribiere=polak_ribiere,
                        est_bound=est_bound, est_step_size=est_step_size, est_window=est_window)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        self.reset_estimate(
            est_bound=est_bound, est_step_size=est_step_size, est_window=est_window,
            polak_ribiere=polak_ribiere, mbound=mbound, interp=interp,ls_prob=ls_prob,
            )
        super(CGQ, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(CGQ, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('polak_ribiere', None)

    def reset_estimate(
        self, est_step_size=None, est_window=None, est_bound=None, 
        polak_ribiere=None, mbound=None, interp=None,ls_prob=None,
    ):
        if est_step_size is not None:
            self.est_step_size = est_step_size
        if est_window is not None:
            self.est_window = est_window # 10
        if est_bound is not None:
            self.est_bound = est_bound
        if polak_ribiere is not None:
            self.polak_ribiere = polak_ribiere
        if mbound is not None:
            self.mbound = mbound
        if interp is not None:
            self.interp = interp
        if ls_prob is not None:
            self.ls_prob = ls_prob
        assert self.interp=='2pt' or (self.interp=='3pt' and self.est_window>1)

    @torch.no_grad()
    def get_dp(self, group, get_grad_only=False):
        ### overall polak-ribiere momentum
        # for group in self.param_groups:
        weight_decay = group['weight_decay']
        momentum = group['momentum']
        dampening = group['dampening']
        nesterov = group['nesterov']
        polak_ribiere = group['polak_ribiere']
        if get_grad_only and polak_ribiere is not None:
            _num = 0.
            _denom = 0.
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                param_state = self.state[p]
                if 'last_grad' in param_state:
                    last_grad = param_state['last_grad']
                    if polak_ribiere == 'FR': ### Fletcher-Reeves
                        _num += d_p.norm()**2
                        _denom += last_grad.norm()**2
                    elif polak_ribiere == 'HS': ### Hestenes-Stiefel
                        diff_dp = (d_p-last_grad).view(-1)
                        _num += d_p.view(-1).dot(diff_dp)
                        _denom += param_state['momentum_buffer'].view(-1).dot(diff_dp)
                    elif polak_ribiere == 'DY': ### Dai–Yuan
                        _num += d_p.norm()**2
                        _denom += param_state['momentum_buffer'].view(-1).dot((d_p-last_grad).view(-1))
                    else: ### Polak-Ribiere
                        _num += d_p.view(-1).dot((d_p-last_grad).view(-1))
                        _denom += last_grad.norm()**2
                            

                ### update last_grad
                param_state['last_grad'] = torch.clone(d_p).detach()
                
            # if _num!=0 and _denom != 0:
            if _denom != 0:
                momentum = torch.clamp(_num/_denom, min=0., max=self.mbound).item()
                group['momentum'] = group['momentum']*0.5 + momentum*0.5
                # group['momentum'] = momentum
        
        slope = 0.
        _norm = 0.
        for p in group['params']:
            if p.grad is None:
                continue
            d_p = p.grad
            param_state = self.state[p]
            if weight_decay != 0:
                d_p = d_p.add(p, alpha=weight_decay)

            if 'momentum_buffer' not in param_state:
                buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
            else:
                buf = param_state['momentum_buffer']

            if momentum != 0:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                if nesterov:
                    d_p = d_p.add(buf, alpha=momentum)
                else:
                    d_p = buf.clone()

            if get_grad_only:
                slope -= p.grad.view(-1).dot(d_p.view(-1))
                # _norm += p.grad.norm().pow(2)
                p.grad.copy_(d_p)
            else:
                p.add_(d_p, alpha=-group['lr'])
        # if get_grad_only:
        #     _norm = np.sqrt(_norm.item())
        return slope, _norm

    @torch.no_grad()
    def step(self, closure=None, loss=None, suffix=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        if loss is None and closure is not None:
            self.zero_grad()
            with torch.enable_grad():
                loss, output = closure(return_out=True)
                loss.backward()

        ### update direction with momentum
        do_line_search = self.ls_prob is not None and random.random()<self.ls_prob
        for group in self.param_groups:
            slope, grad_norm = self.get_dp(group, do_line_search)
            if do_line_search:
                ### find optimal learning rate
                _lr = opt_lr(group, closure, loss, slope=slope.item(), grad_norm=grad_norm, method=self.interp, plot=False, debug=False, 
                            suffix=suffix, est_window=self.est_window, est_bound=self.est_bound, est_step_size=self.est_step_size)
                ### adjust optimizer learning rate
                group['lr'] = group['lr']*0.5 + _lr*0.5
        return loss, output

#########################################################################
######################### Armijo Line Search #########################
#########################################################################
# reference: https://github.com/IssamLaradji/sls

def check_armijo_conditions(step_size, step_size_old, loss, grad_norm,
                      loss_next, c, beta_b):
    found = 0
    # computing the new break condition
    break_condition = loss_next - \
        (loss - (step_size) * c * grad_norm**2)
    if (break_condition <= 0):
        found = 1

    else:
        # decrease the step-size by a multiplicative factor
        step_size = step_size * beta_b
        
    return found, step_size, step_size_old

def check_goldstein_conditions(step_size, loss, grad_norm,
                          loss_next,
                          c, beta_b, beta_f, bound_step_size, eta_max):
	found = 0
	if(loss_next <= (loss - (step_size) * c * grad_norm ** 2)):
		found = 1

	if(loss_next >= (loss - (step_size) * (1 - c) * grad_norm ** 2)):
		if found == 1:
			found = 3 # both conditions are satisfied
		else:
			found = 2 # only the curvature condition is satisfied

	if (found == 0):
		raise ValueError('Error')

	elif (found == 1):
		# step-size might be too small
		step_size = step_size * beta_f
		if bound_step_size:
			step_size = min(step_size, eta_max)

	elif (found == 2):
		# step-size might be too large
		step_size = max(step_size * beta_b, 1e-8)

	return {"found":found, "step_size":step_size}


def reset_step(step_size, n_batches_per_epoch=None, gamma=None, reset_option=1,
               init_step_size=None):
    if reset_option == 0:
        pass

    elif reset_option == 1:
        step_size = step_size * gamma**(1. / n_batches_per_epoch)

    elif reset_option == 2:
        step_size = init_step_size

    return step_size

def try_sgd_update(params, step_size, params_current, grad_current):
    zipped = zip(params, params_current, grad_current)

    for p_next, p_current, g_current in zipped:
        p_next.data = p_current - step_size * g_current

def compute_grad_norm(grad_list):
    grad_norm = 0.
    for g in grad_list:
        if g is None:
            continue
        grad_norm += torch.sum(torch.mul(g, g))
    grad_norm = torch.sqrt(grad_norm)
    return grad_norm


def get_grad_list(params):
    return [p.grad for p in params]

@contextlib.contextmanager
def random_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

@contextlib.contextmanager
def random_seed_torch(seed, device=0):
    cpu_rng_state = torch.get_rng_state()
    gpu_rng_state = torch.cuda.get_rng_state(0)

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    try:
        yield
    finally:
        torch.set_rng_state(cpu_rng_state)
        torch.cuda.set_rng_state(gpu_rng_state, device)


class Sls(torch.optim.Optimizer):
    """Implements stochastic line search
    `paper <https://arxiv.org/abs/1905.09997>`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        n_batches_per_epoch (int, recommended):: the number batches in an epoch
        init_step_size (float, optional): initial step size (default: 1)
        c (float, optional): armijo condition constant (default: 0.1)
        beta_b (float, optional): multiplicative factor for decreasing the step-size (default: 0.9)
        gamma (float, optional): factor used by Armijo for scaling the step-size at each line-search step (default: 2.0)
        beta_f (float, optional): factor used by Goldstein for scaling the step-size at each line-search step (default: 2.0)
        reset_option (float, optional): sets the rest option strategy (default: 1)
        eta_max (float, optional): an upper bound used by Goldstein on the step size (default: 10)
        bound_step_size (bool, optional): a flag used by Goldstein for whether to bound the step-size (default: True)
        line_search_fn (float, optional): the condition used by the line-search to find the 
                    step-size (default: Armijo)
    """

    def __init__(self,
                 params,
                 n_batches_per_epoch=500,
                 init_step_size=1,
                 c=0.1,
                 beta_b=0.9,
                 gamma=2.0,
                 beta_f=2.0,
                 reset_option=1,
                 eta_max=10,
                 bound_step_size=True,
                 line_search_fn="armijo"):
        defaults = dict(n_batches_per_epoch=n_batches_per_epoch,
                        init_step_size=init_step_size,
                        c=c,
                        beta_b=beta_b,
                        gamma=gamma,
                        beta_f=beta_f,
                        reset_option=reset_option,
                        eta_max=eta_max,
                        bound_step_size=bound_step_size,
                        line_search_fn=line_search_fn)
        super().__init__(params, defaults)       

        self.state['step'] = 0
        self.state['step_size'] = init_step_size

        self.state['n_forwards'] = 0
        self.state['n_backwards'] = 0

    def step(self, closure):
        # deterministic closure
        seed = time.time()
        def closure_deterministic(**kwargs):
            with random_seed_torch(int(seed)):
                return closure(**kwargs)

        batch_step_size = self.state['step_size']

        # get loss and compute gradients
        loss,output = closure_deterministic(return_out=True)
        loss.backward()

        # increment # forward-backward calls
        self.state['n_forwards'] += 1
        self.state['n_backwards'] += 1

        # loop over parameter groups
        for group in self.param_groups:
            params = group["params"]

            # save the current parameters:
            params_current = copy.deepcopy(params)
            grad_current = get_grad_list(params)

            grad_norm = compute_grad_norm(grad_current)

            step_size = reset_step(step_size=batch_step_size,
                                    n_batches_per_epoch=group['n_batches_per_epoch'],
                                    gamma=group['gamma'],
                                    reset_option=group['reset_option'],
                                    init_step_size=group['init_step_size'])
            temp_lr = 0.
            # only do the check if the gradient norm is big enough
            with torch.no_grad():
                if grad_norm >= 1e-8:
                    # check if condition is satisfied
                    found = 0
                    step_size_old = step_size

                    for e in range(100):
                        # try a prospective step
                        try_sgd_update(params, step_size, params_current, grad_current)
                        temp_lr = step_size

                        # compute the loss at the next step; no need to compute gradients.
                        loss_next = closure_deterministic()
                        self.state['n_forwards'] += 1

                        # =================================================
                        # Line search
                        if group['line_search_fn'] == "armijo":
                            armijo_results = check_armijo_conditions(step_size=step_size,
                                                        step_size_old=step_size_old,
                                                        loss=loss,
                                                        grad_norm=grad_norm,
                                                        loss_next=loss_next,
                                                        c=group['c'],
                                                        beta_b=group['beta_b'])
                            found, step_size, step_size_old = armijo_results
                            if found == 1:
                                break
                        
                        elif group['line_search_fn'] == "goldstein":
                            goldstein_results = check_goldstein_conditions(step_size=step_size,
                                                                    loss=loss,
                                                                    grad_norm=grad_norm,
                                                                    loss_next=loss_next,
                                                                    c=group['c'],
                                                                    beta_b=group['beta_b'],
                                                                    beta_f=group['beta_f'],
                                                                    bound_step_size=group['bound_step_size'],
                                                                    eta_max=group['eta_max'])

                            found = goldstein_results["found"]
                            step_size = goldstein_results["step_size"]

                            if found == 3:
                                break
                
                    # if line search exceeds max_epochs
                    if found == 0:
                        try_sgd_update(params, 1e-6, params_current, grad_current)
                        temp_lr = 1e-6

            # save the new step-size
            self.state['step_size'] = step_size
            self.state['lr'] = temp_lr
            self.state['step'] += 1

        return loss,output


#########################################################################
######################### Lookahead Optimizer #########################
#########################################################################

class Lookahead(Optimizer):
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        for group in self.param_groups:
            group["counter"] = 0
    
    def update(self, group):
        for fast in group["params"]:
            param_state = self.state[fast]
            if "slow_param" not in param_state:
                param_state["slow_param"] = torch.zeros_like(fast.data)
                param_state["slow_param"].copy_(fast.data)
            slow = param_state["slow_param"]
            slow += (fast.data - slow) * self.alpha
            fast.data.copy_(slow)
    
    def update_lookahead(self):
        for group in self.param_groups:
            self.update(group)

    def step(self, closure=None, **kwargs):
        loss = self.optimizer.step(closure, **kwargs)
        for group in self.param_groups:
            if group["counter"] == 0:
                self.update(group)
            group["counter"] += 1
            if group["counter"] >= self.k:
                group["counter"] = 0
        return loss

    def state_dict(self):
        fast_state_dict = self.optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict["state"]
        param_groups = fast_state_dict["param_groups"]
        return {
            "fast_state": fast_state,
            "slow_state": slow_state,
            "param_groups": param_groups,
        }

    def load_state_dict(self, state_dict):
        slow_state_dict = {
            "state": state_dict["slow_state"],
            "param_groups": state_dict["param_groups"],
        }
        fast_state_dict = {
            "state": state_dict["fast_state"],
            "param_groups": state_dict["param_groups"],
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.optimizer.load_state_dict(fast_state_dict)
        self.fast_state = self.optimizer.state

    def add_param_group(self, param_group):
        param_group["counter"] = 0
        self.optimizer.add_param_group(param_group)


#########################################################################
######################### Coin Betting Optimizer ########################
#########################################################################
class COCOB_Backprop(Optimizer):
    """Implementation of the COCOB algorithm.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    
    Usage:
    1. Put cocob_bp.py in YOUR_PYTHON_PATH/site-packages/torch/optim. 
    2. Open YOUR_PYTHON_PATH/site-packages/torch/optim/__init__.py add the following code:
    ```
    from .cocob_bp import COCOB_Backprop
    del cocob_bp
    ```
    3. Save __init__.py and restart your python. 
    Use COCOB_Backprop as

    optimizer = optim.COCOB_Backprop(net.parameters())
    ...
    optimizer.step()

    Implemented by Huidong Liu
    Email: huidliu@cs.stonybrook.edu or h.d.liew@gmail.com

    References
    [1] Francesco Orabona and Tatiana Tommasi, Training Deep Networks without Learning Rates
    Through Coin Betting, NIPS 2017.
    
    """
    def __init__(self, params, weight_decay=0, alpha=100):
        defaults = dict(weight_decay=weight_decay)
        super(COCOB_Backprop, self).__init__(params, defaults)
        # COCOB initializaiton
        self.W1 = []
        self.W_zero = []
        self.W_one = []
        self.L = []
        self.G = []
        self.Reward = []
        self.Theta = []
        self.numPara = 0
        self.weight_decay = weight_decay
        self.alpha = alpha
        
        for group in self.param_groups:
            for p in group['params']:
                self.W1.append(p.data.clone())
                self.W_zero.append(p.data.clone().zero_())
                self.W_one.append(p.data.clone().fill_(1))
                self.L.append(p.data.clone().fill_(1))
                self.G.append(p.data.clone().zero_())
                self.Reward.append(p.data.clone().zero_())
                self.Theta.append(p.data.clone().zero_())                
                self.numPara = self.numPara + 1     
            

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        pind = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data + self.weight_decay*p.data
                self.L[pind] = self.L[pind].max(grad.abs())
                self.G[pind] = self.G[pind] + grad.abs()
                self.Reward[pind] = (self.Reward[pind]-(p.data-self.W1[pind]).mul(grad)).max(self.W_zero[pind])
                self.Theta[pind] = self.Theta[pind] + grad
                Beta = self.Theta[pind].div( (self.alpha*self.L[pind]).max(self.G[pind]+self.L[pind]) ).div(self.L[pind])
                p.data = self.W1[pind] - Beta.mul(self.L[pind] + self.Reward[pind])
                pind = pind + 1
                
        return loss





#########################################################################
######################### PAL Optimizer #########################
#########################################################################

class _RequiredParameter(object):
    """Singleton class representing a required parameter for an Optimizer."""

    def __repr__(self):
        return "<required parameter>"


required = _RequiredParameter()


class PalOptimizer(Optimizer):
    def __init__(self, params=required, writer=None, measuring_step_size=1, max_step_size=3.16,
                 direction_adaptation_factor=0.4, update_step_adaptation=1 / 0.6,
                 epsilon=1e-10, calc_exact_directional_derivative=True, is_plot=False, plot_step_interval=100,
                 save_dir="/tmp/lines/"):
        """
        The PAL optimizer.
        Approximates the loss in negative gradient direction with a one-dimensional parabolic function.
        Uses the location of the minimum of the approximation for weight updates.
        :param params: net.parameters()
        :param writer: optional tensorboardX writer for detailed logs
        :param measuring_step_size: Good values are between 0.1 and 1
        :param max_step_size:  Good values are between 1 and 10. Low sensitivity.
        :param direction_adaptation_factor. Good values are either 0 or 0.4. Low sensitivity.
        :param update_step_adaptation: loose approximation term. Good values are between 1.2 and 1.7. Low sensitivity.
        :param calc_exact_directional_derivative: more exact approximation but more time consuming
        :param is_plot: plot loss line and approximation
        :param plot_step_interval: training_step % plot_step_interval == 0 -> plot the line the approximation is done over
        :param save_dir: line plot save location
        """

        if is_plot == True and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if measuring_step_size <= 0.0:
            raise ValueError("Invalid measuring step size: {}".format(measuring_step_size))
        if max_step_size < 0.0:
            raise ValueError("Invalid measuring maximal step size: {}".format(max_step_size))
        if direction_adaptation_factor < 0.0:
            raise ValueError("Invalid measuring direction_adaptation_factor: {}".format(direction_adaptation_factor))
        if update_step_adaptation < 0.0:
            raise ValueError("Invalid loose approximation factor: {}".format(update_step_adaptation))
        if plot_step_interval < 1 or plot_step_interval % 1 is not 0:
            raise ValueError("Invalid plot_step_interval factor: {}".format(plot_step_interval))

        if measuring_step_size is not type(torch.Tensor):
            measuring_step_size = torch.tensor(measuring_step_size)
        if max_step_size is not type(torch.Tensor):
            max_step_size = torch.tensor(max_step_size)
        if direction_adaptation_factor is not type(torch.Tensor):
            direction_adaptation_factor = torch.tensor(direction_adaptation_factor)
        if update_step_adaptation is not type(torch.Tensor):
            update_step_adaptation = torch.tensor(update_step_adaptation)

        self.writer = writer
        self.train_steps = -1
        self.time_start = time.time()
        defaults = dict(measuring_step_size=measuring_step_size,
                        max_step_size=max_step_size, direction_adaptation_factor=direction_adaptation_factor,
                        update_step_adaptation=update_step_adaptation, epsilon=epsilon,
                        calc_exact_directional_derivative=calc_exact_directional_derivative, is_plot=is_plot,
                        plot_step_interval=plot_step_interval, save_dir=save_dir)
        super(PalOptimizer, self).__init__(params, defaults)

    def _set_momentum_get_norm_and_derivative(self, params, direction_adaptation_factor, epsilon,
                                              calc_exact_directional_derivative):
        """ applies direction_adaptation_factor to the gradients and saves result in param state cg_buffer """
        with torch.no_grad():
            directional_derivative = torch.tensor(0.0)
            norm = torch.tensor(0.0)
            if direction_adaptation_factor != 0:
                for p in params:
                    if p.grad is None:
                        continue
                    param_state = self.state[p]
                    if 'cg_buffer' not in param_state:
                        buf = param_state['cg_buffer'] = torch.zeros_like(p.grad.data,device=p.device)
                    else:
                        buf = param_state['cg_buffer']
                    buf = buf.mul_(direction_adaptation_factor)
                    buf = buf.add_(p.grad.data)
                    flat_buf = buf.view(-1)
                    flat_grad = p.grad.data.view(-1)
                    if calc_exact_directional_derivative is True:
                        directional_derivative = directional_derivative + torch.dot(flat_grad, flat_buf)
                    norm = norm + torch.dot(flat_buf, flat_buf)
                    p.grad.data = buf.clone()
                norm = torch.sqrt(norm)
                if norm == 0: norm = epsilon
                if calc_exact_directional_derivative is True:
                    directional_derivative = - directional_derivative / norm
                else:
                    directional_derivative = -norm
            else:
                for p in params:
                    if p.grad is None:
                        continue
                    flat_grad = p.grad.data.view(-1)
                    norm = norm + torch.dot(flat_grad, flat_grad)
                norm = torch.sqrt(norm)
                if norm == 0: norm = epsilon
                directional_derivative = -norm

        return norm, directional_derivative

    def _perform_param_update_step(self, params, step, direction_norm):
        """ SGD-like update step of length 'measuring_step_size' in negative gradient direction """
        with torch.no_grad():
            if step != 0:
                for p in params:
                    if p.grad is None:
                        continue
                    param_state = self.state[p]
                    if 'cg_buffer' in param_state:
                        line_direction = param_state['cg_buffer']
                        p.data.add_(step * -line_direction / direction_norm)
                    else:
                        p.data.add_(step * -p.grad.data / direction_norm)

    def step(self, loss_fn):
        """
        Performs a PAL optimization step,
        calls the loss_fn twice
        E.g.:
        >>> def loss_fn(backward=True):
        >>>     out_ = net(inputs)
        >>>     loss_ = criterion(out_, targets)
        >>>     if backward:
        >>>         loss_.backward()
        >>> return loss_, out_
        :param loss_fn: function that returns the loss as the first output
                        requires 2 or more return values, e.g. also result of the forward pass
                        requires a backward parameter, whether a backward pass is required or not
                        the loss has to be backpropagated when backward is set to True
        :return: outputs of the first loss_fn call and the estimated step size
        """
        seed = time.time()

        def loss_fn_deterministic(backward=True):
            with self.random_seed_torch(int(seed)):
                return loss_fn(backward)

        self.train_steps += 1
        with torch.no_grad():  #
            for group in self.param_groups:
                params = group['params']
                measuring_step = group['measuring_step_size']
                max_step_size = group['max_step_size']
                update_step_adaptation = group['update_step_adaptation']
                direction_adaptation_factor = group['direction_adaptation_factor']
                epsilon = group['epsilon']
                is_plot = group['is_plot']
                plot_step_interval = group['plot_step_interval']
                save_dir = group['save_dir']
                calc_exact_directional_derivative = group['calc_exact_directional_derivative']

                # get gradients for each param
                with torch.enable_grad():
                    loss_0, returns = loss_fn_deterministic(backward=True)
                direction_norm, directional_derivative = self._set_momentum_get_norm_and_derivative(params,
                                                                                                    direction_adaptation_factor,
                                                                                                    epsilon,
                                                                                                    calc_exact_directional_derivative)

                # sample step of length measuring_step_size
                self._perform_param_update_step(params, measuring_step, direction_norm)
                loss_mu, *_ = loss_fn_deterministic(backward=False)

                # parabolic parameters
                b = directional_derivative
                a = (loss_mu - loss_0 - directional_derivative * measuring_step) / (measuring_step ** 2)
                # c = loss_0

                if torch.isnan(a) or torch.isnan(b) or torch.isinf(a) or torch.isinf(b):
                    return loss_0 , returns, 0.0

                # get jump distance
                if a > 0 and b < 0:
                    s_upd = -b / (2 * a) * update_step_adaptation
                elif a <= 0 and b < 0:
                    s_upd = measuring_step.clone()  # clone() since otherwise it's a reference to the measuring_step object
                else:
                    s_upd = torch.tensor(0.0)

                if s_upd > max_step_size:
                    s_upd = max_step_size.clone()
                s_upd -= measuring_step
                group['lr'] = (s_upd/direction_norm).item()


                #### plotting
                if is_plot and self.train_steps % plot_step_interval == 0:
                    self.plot_loss_line_and_approximation(measuring_step / 20, s_upd, measuring_step, direction_norm,
                                                          loss_fn_deterministic, a, b, loss_0, loss_mu, params,
                                                          save_dir)

                # log some info, via batch and time[ms]
                if self.writer is not None:
                    cur_time = int((time.time() - self.time_start) * 1000)  # log in ms since it has to be an integer
                    for s, t in [('time', cur_time), ('batch', self.train_steps)]:
                        self.writer.add_scalar('train-%s/l_0' % s, loss_0.item(), t)
                        self.writer.add_scalar('train-%s/l_mu' % s, loss_mu.item(), t)
                        self.writer.add_scalar('train-%s/b' % s, b.item(), t)
                        self.writer.add_scalar('train-%s/a' % s, a.item(), t)
                        self.writer.add_scalar('train-%s/measuring_step_size' % s, measuring_step, t)
                        self.writer.add_scalar('train-%s/mss' % s, max_step_size, t)
                        self.writer.add_scalar('train-%s/s_upd' % s, s_upd, t)
                        self.writer.add_scalar('train-%s/grad_norm' % s, direction_norm.item(), t)

                self._perform_param_update_step(params, s_upd, direction_norm)

                return loss_0, returns ,((s_upd+measuring_step)/direction_norm).item()

    def plot_loss_line_and_approximation(self, resolution, a_min, mu, direction_norm, loss_fn, a, b, loss_0, loss_mu,
                                         params,
                                         save_dir):
        resolution = resolution.clone()
        a_min = a_min.clone()
        mu = mu.clone()
        direction_norm = direction_norm.clone()
        a = a.clone()
        b = b.clone()
        loss_0 = loss_0.clone()
        loss_mu = loss_mu.clone()

        # parabola parameters:
        a = a.detach().cpu().numpy()
        b = b.detach().cpu().numpy()
        c = loss_0.detach().cpu().numpy()

        real_a_min = (a_min + mu).detach().cpu().numpy()
        line_losses = []
        resolution = resolution * 2
        resolution_v = (resolution).detach().cpu().numpy()
        max_step = 2
        min_step = 1
        interval = list(np.arange(-2 * resolution_v - min_step, max_step + 2 * resolution_v, resolution_v))
        self._perform_param_update_step(params, -mu - 2 * resolution - min_step, direction_norm)
        line_losses.append(loss_fn(backward=False)[0].detach().cpu().numpy())

        for i in range(len(interval) - 1):
            self._perform_param_update_step(params, resolution, direction_norm)
            line_losses.append(loss_fn(backward=False)[0].detach().cpu().numpy())

        def parabolic_function(x, a, b, c):
            """
            :return:  value of f(x)= a(x-t)^2+b(x-t)+c
            """
            return a * x ** 2 + b * x + c

        x = interval
        x2 = list(np.arange(-resolution_v, 1.1 * resolution_v, resolution_v))

        plt.rc('text', usetex=True)
        plt.rc('font', serif="Times")
        scale_factor = 1
        tick_size = 23 * scale_factor
        label_size = 23 * scale_factor
        heading_size = 26 * scale_factor
        fig_sizes = np.array([10, 8]) * scale_factor

        fig = plt.figure(0)
        fig.set_size_inches(fig_sizes)
        plt.plot(x, line_losses, linewidth=3.0)
        approx_values = [parabolic_function(x_i, a, b, c) for x_i in x]
        plt.plot(x, approx_values, linewidth=3.0)
        grad_values = [b * x2_i + c for x2_i in x2]
        plt.plot(x2, grad_values, linewidth=3.0)
        plt.axvline(real_a_min, color="red", linewidth=3.0)
        y_max = max(line_losses)
        y_min = min(min(approx_values), min(line_losses))
        plt.ylim([y_min, y_max])
        plt.legend(["loss", "approximation", "derivative", r"$s_{min}$"], fontsize=label_size)
        plt.xlabel("step on line", fontsize=label_size)
        plt.ylabel("loss in line direction", fontsize=label_size)
        plt.plot(0, c, 'x')

        mu_v = mu.detach().cpu().numpy()
        loss_mu_v = loss_mu.detach().cpu().numpy()
        plt.plot(mu_v, loss_mu_v, 'x')

        global_step = self.train_steps
        plt.title("Loss line of step {0:d}".format(global_step), fontsize=heading_size)

        plt.gca().tick_params(
            axis='both',
            which='both',
            labelsize=tick_size)
        plt.show(block=True)
        plt.savefig("{0}line{1:d}.png".format(save_dir, global_step))
        print("plotted line {0}line{1:d}.png".format(save_dir, global_step))
        #plt.show(block=True)
        plt.close(0)
        positive_steps = sum(i > 0 for i in interval)
        self._perform_param_update_step(params, - positive_steps * resolution + mu, direction_norm)

    @contextlib.contextmanager
    def random_seed_torch(self, seed):
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