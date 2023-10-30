import math

import torch
from torch.optim.optimizer import Optimizer


class IPalm(Optimizer):
    r"""Implements inertial proximal alternatinglinearizing minimization
    algorithm.

    It has been proposed in `Inertial Proximal Alternating Linearized
    Minimization (iPALM) for Nonconvex and Nonsmooth Problems`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): initial learning rate (default: 1e-3)
        beta (float, optional): momentum coefficients (default: 1/math.sqrt(2))
        eps (float, optional): tolerance in % in the back tracking step
            (default: 1e-3%)
        nb (int, optional): number of backtracking steps (default: 20)

    .. _IPALM\:https://epubs.siam.org/doi/abs/10.1137/16M1064064
    """
    def __init__(
        self,
        params,
        lr=1e-3,
        beta=1 / math.sqrt(2),
        eps=1e-5,
        nb=30,
        use_precond=False,
        beta2=0.99
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= beta < 1.0:
            raise ValueError("Invalid beta parameter: {}".format(beta))
        defaults = dict(
            lr=lr,
            beta=beta,
            eps=eps,
            nb=nb,
            use_precond=use_precond,
            beta2=beta2
        )
        super(IPalm, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(IPalm, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            beta = group['beta']
            beta2 = group['beta2']
            # initialize and overrelax parameters in group
            for p in group['params']:

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # old variable value
                    state['tilde'] = torch.zeros_like(p.data)
                    # old variable value
                    state['old'] = p.data.clone()
                    # Exponential moving average of squared gradient values
                    if group['use_precond']:
                        if hasattr(p, 'reduction_dim'):
                            state['grad_norm'] = torch.zeros_like(
                                torch.sum(p.data, p.reduction_dim, True)
                            )
                        else:
                            state['grad_norm'] = torch.zeros_like(p.data)
                        state['D'] = torch.ones_like(state['grad_norm'])

                state['step'] += 1

                # overrelaxation
                p_old = p.data.clone()
                p.data.mul_(1 + beta).sub_(state['old'], alpha=beta)
                state['old'].copy_(p_old)
                state['tilde'].copy_(p.data)

            # compute the loss and the gradient at the overrelaxed position
            self.zero_grad()
            # with torch.autograd.detect_anomaly():
            losses = closure(compute_grad=True)
            loss = losses[0]
            loss.backward()
            # distribute the gradient TODO: only to leave nodes that are required!

            with torch.no_grad():
                # update the preconditioning
                if group['use_precond']:
                    for p in group['params']:
                        if hasattr(p, 'reduction_dim'):
                            self.state[p]['grad_norm'].mul_(beta2).add_(
                                1 - beta2,
                                torch.sum(p.grad**2, p.reduction_dim, True)
                            )
                        else:
                            self.state[p]['grad_norm'].mul_(beta2).add_(
                                1 - beta2, p.grad**2
                            )
                        self.state[p]['D'] = self.state[p]['grad_norm'].sqrt(
                        ).add_(1e-5)

                    bias_correction = math.sqrt(1 - beta2**state['step'])

                for j in range(group['nb']):
                    # update all parameters in the group
                    for p in group['params']:
                        if group['use_precond']:
                            p.data = self.state[p][
                                'tilde'] - group['lr'] * p.grad / self.state[
                                    p]['D'] * bias_correction
                        else:
                            p.data = self.state[p]['tilde'
                                                   ] - group['lr'] * p.grad
                        if hasattr(p, 'proj'):
                            p.proj()
                        if hasattr(p, 'prox'):
                            p.prox(group['lr'])

                    # reevaluate the loss
                    losses_new = closure()
                    loss_new = losses_new[0]

                    # compute the quadratic upper bound
                    t1 = 0
                    t2 = 0
                    for p in group['params']:
                        t1 += (p.grad *
                               (p.data - self.state[p]['tilde'])).sum()
                        if group['use_precond']:
                            t2 += ((p.data - self.state[p]['tilde'])**2 *
                                   self.state[p]['D']).sum(
                                   ) / (2 * group['lr'] * bias_correction)
                        else:
                            t2 += ((p.data - self.state[p]['tilde'])**
                                   2).sum() / (2 * group['lr'])
                    bound = loss + t1 + t2

                    delta = 1 + torch.sign(bound) * group['eps']
                    if loss_new < bound * delta:
                        group['lr'] = min(group['lr'] * 2, 1e+8)
                        break
                    else:
                        group['lr'] = max(group['lr'] / 2, 1e-20)
                        # print(group['name'], 'bt={:2d} loss_new={} loss_old={} bound={} t1={} t2={} ||g||={}'.format(
                        #     j, loss_new.item(), loss.item(), bound.item(),
                        #     t1.item(), t2.item(),
                        #     (p.grad**2).sum().cpu().numpy()
                        #     ), group['lr'])
                        if group['lr'] == 1e-15:
                            break

                print(
                    '{:15s} lr={:.3e}, E={:.3e}, Q={:.3e}'.format(
                        group['name'], group['lr'], loss_new.item(),
                        bound.item()
                    )
                )

        return losses_new
