import torch
from torch.optim.optimizer import Optimizer

class Sophia(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), rho=0.03,
                 weight_decay=0.0, k=10, hessian_computation_type='gnb',
                 eps=1e-12):
        if not 0.0 <= lr: raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps: raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0: raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0: raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= rho: raise ValueError(f"Invalid rho value: {rho}")
        if not 0.0 <= weight_decay: raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, rho=rho, weight_decay=weight_decay, k=k, hessian_computation_type=hessian_computation_type, eps=eps)
        super(Sophia, self).__init__(params, defaults)

    def _compute_hessian_gnb(self, logits, p):
        with torch.enable_grad():
            probs = torch.softmax(logits, dim=-1)
            sampled_labels = torch.multinomial(probs, num_samples=1).squeeze(-1)
            fake_loss = torch.nn.functional.cross_entropy(logits, sampled_labels, reduction='mean')
            fake_grad = torch.autograd.grad(fake_loss, p, retain_graph=True)[0]
        return fake_grad.pow(2)

    @torch.no_grad()
    def step(self, closure=None, logits=None):
        loss = None
        grad_map = {}
        hessian_map = {}

        # --- Phase 1: Gradient and Hessian Calculation ---
        if self.defaults['hessian_computation_type'] == 'hutchinson':
            with torch.enable_grad():
                loss = closure()
                params_with_grad = [p for group in self.param_groups for p in group['params'] if p.requires_grad]
                
                first_order_grads = torch.autograd.grad(loss, params_with_grad, create_graph=True)
                
                V = [torch.randn_like(p) for p in params_with_grad]
                grad_v_sum = sum((g * v).sum() for g, v in zip(first_order_grads, V))
                hessian_vector_products = torch.autograd.grad(grad_v_sum, params_with_grad, retain_graph=False)
                
                for i, p in enumerate(params_with_grad):
                    grad_map[p] = first_order_grads[i]
                    hessian_map[p] = (V[i] * hessian_vector_products[i]).abs()

        # --- Phase 2: Update Calculation Loop ---
        updates = []
        for group in self.param_groups:
            for p in group['params']:
                grad = grad_map.get(p, p.grad)
                if grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum_buffer'] = torch.zeros_like(p)
                    state['hessian_buffer'] = torch.zeros_like(p)

                state['step'] += 1
                beta1, beta2 = group['betas']
                
                momentum = state['momentum_buffer']
                momentum.mul_(beta1).add_(grad, alpha=1. - beta1)

                if state['step'] % group['k'] == 0:
                    hessian_estimator = None
                    if group['hessian_computation_type'] == 'gnb':
                        hessian_estimator = self._compute_hessian_gnb(logits, p)
                    else: # hutchinson
                        hessian_estimator = hessian_map[p]
                    
                    hessian_buffer = state['hessian_buffer']
                    hessian_buffer.mul_(beta2).add_(hessian_estimator, alpha=1. - beta2)

                hessian_buffer = state['hessian_buffer']
                denominator = hessian_buffer + group['eps']
                update = torch.clamp(momentum / denominator, min=-group['rho'], max=group['rho'])
                updates.append((p, update, group))

        # --- Phase 3: Apply Updates ---
        for p, update, group in updates:
            if group['weight_decay'] > 0.0:
                p.mul_(1. - group['weight_decay'])
            p.add_(update, alpha=-group['lr'])
        
        return loss
