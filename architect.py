""" Architect controls architecture of cell by computing gradients of alphas """
import copy
import torch


class Architect():
    """ Compute gradients of alphas """
    def __init__(self, net_1, net_2, w_momentum, w_weight_decay):
        """
        Args:
            net
            w_momentum: weights momentum
        """
        self.net_1 = net_1
        self.net_2 = net_2
        self.v_net_1 = copy.deepcopy(net_1)
        self.v_net_2 = copy.deepcopy(net_2)
        self.w_momentum = w_momentum
        self.w_weight_decay = w_weight_decay

    def virtual_step(self, trn_X, trn_y, xi_1, xi_2, w_optim_1, w_optim_2, lmbda):
        """
        Compute unrolled weight w' (virtual step)

        Step process:
        1) forward
        2) calc loss
        3) compute gradient (by backprop)
        4) update gradient

        Args:
            xi: learning rate for virtual gradient step (same as weights lr)
            w_optim: weights optimizer
        """
        # forward & calc loss
        logits_1 = self.net_1.forward(trn_X)
        pseudolabel_1 = torch.argmax(logits_1, dim=1)
        logits_2 = self.net_2.forward(trn_X)
        pseudolabel_2 = torch.argmax(logits_2, dim=1)
        loss = self.net_1.loss(trn_X, trn_y) + self.net_2.loss(trn_X, trn_y) + lmbda * self.net_1.loss(trn_X, pseudolabel_2) + lmbda * self.net_2.loss(trn_X, pseudolabel_1) # L_trn(w)

        # compute gradient
        gradients_1 = torch.autograd.grad(loss, self.net_1.weights())
        gradients_2 = torch.autograd.grad(loss, self.net_2.weights())


        # do virtual step (update gradient)
        # below operations do not need gradient tracking
        with torch.no_grad():
            # dict key is not the value, but the pointer. So original network weight have to
            # be iterated also.
            for w, vw, g in zip(self.net_1.weights(), self.v_net_1.weights(), gradients_1):
                m = w_optim_1.state[w].get('momentum_buffer', 0.) * self.w_momentum
                vw.copy_(w - xi_1 * (m + g + self.w_weight_decay*w))
            for w, vw, g in zip(self.net_2.weights(), self.v_net_2.weights(), gradients_2):
                m = w_optim_2.state[w].get('momentum_buffer', 0.) * self.w_momentum
                vw.copy_(w - xi_2 * (m + g + self.w_weight_decay*w))

            # synchronize alphas
            for a, va in zip(self.net_1.alphas(), self.v_net_1.alphas()):
                va.copy_(a)
            for a, va in zip(self.net_2.alphas(), self.v_net_2.alphas()):
                va.copy_(a)

    def unrolled_backward(self, trn_X, trn_y, val_X, val_y, xi_1, xi_2, w_optim_1, w_optim_2, lmbda):
        """ Compute unrolled loss and backward its gradients
        Args:
            xi: learning rate for virtual gradient step (same as net lr)
            w_optim: weights optimizer - for virtual step
        """
        # do virtual step (calc w`)
        self.virtual_step(trn_X, trn_y, xi_1, xi_2, w_optim_1, w_optim_2, lmbda)

        # calc unrolled loss
        loss = self.v_net_1.loss(val_X, val_y) + self.v_net_2.loss(val_X, val_y) # L_val(w`)

        # compute gradient
        v_alphas_1 = tuple(self.v_net_1.alphas())
        v_weights_1 = tuple(self.v_net_1.weights())
        
        v_alphas_2 = tuple(self.v_net_2.alphas())
        v_weights_2 = tuple(self.v_net_2.weights())
        
        v_grads_1 = torch.autograd.grad(loss, v_alphas_1 + v_weights_1)
        v_grads_2 = torch.autograd.grad(loss, v_alphas_2 + v_weights_2)
        
        dalpha_1 = v_grads_1[:len(v_alphas_1)]
        dw_1 = v_grads_1[len(v_alphas_1):]
        
        dalpha_2 = v_grads_2[:len(v_alphas_2)]
        dw_2 = v_grads_2[len(v_alphas_2):]

        hessian_1 = self.compute_hessian(self.net_1, dw_1, trn_X, trn_y)
        hessian_2 = self.compute_hessian(self.net_2, dw_2, trn_X, trn_y)

        # update final gradient = dalpha - xi*hessian
        with torch.no_grad():
            for alpha, da, h in zip(self.net_1.alphas(), dalpha_1, hessian_1):
                alpha.grad = da - xi_1*h
                
        with torch.no_grad():
            for alpha, da, h in zip(self.net_2.alphas(), dalpha_2, hessian_2):
                alpha.grad = da - xi_2*h

    def compute_hessian(self, net, dw, trn_X, trn_y):
        """
        dw = dw` { L_val(w`, alpha) }
        w+ = w + eps * dw
        w- = w - eps * dw
        hessian = (dalpha { L_trn(w+, alpha) } - dalpha { L_trn(w-, alpha) }) / (2*eps)
        eps = 0.01 / ||dw||
        """
        norm = torch.cat([w.view(-1) for w in dw]).norm()
        eps = 0.01 / norm

        # w+ = w + eps*dw`
        with torch.no_grad():
            for p, d in zip(net.weights(), dw):
                p += eps * d
        loss = net.loss(trn_X, trn_y)
        dalpha_pos = torch.autograd.grad(loss, net.alphas()) # dalpha { L_trn(w+) }

        # w- = w - eps*dw`
        with torch.no_grad():
            for p, d in zip(net.weights(), dw):
                p -= 2. * eps * d
        loss = net.loss(trn_X, trn_y)
        dalpha_neg = torch.autograd.grad(loss, net.alphas()) # dalpha { L_trn(w-) }

        # recover w
        with torch.no_grad():
            for p, d in zip(net.weights(), dw):
                p += eps * d

        hessian = [(p-n) / (2.*eps) for p, n in zip(dalpha_pos, dalpha_neg)]
        return hessian
