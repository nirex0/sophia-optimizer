
# A Flexible PyTorch Implementation of the Sophia Optimizer

[](https://www.google.com/search?q=%5Bhttps://opensource.org/licenses/MIT%5D\(https://opensource.org/licenses/MIT\))

This repository contains a from-scratch, educational implementation of the **Sophia** optimizer in PyTorch. This implementation was developed and rigorously debugged to be flexible and robust, supporting both classification and general-purpose regression tasks out-of-the-box.

The core of this project is a single, unified `Sophia` class that can be configured to use:

  * **Sophia-G**: The fast Gauss-Newton-Bartlett (GNB) estimator for tasks using Cross-Entropy loss.
  * **Sophia-H**: The general-purpose Hutchinson's estimator for any arbitrary, twice-differentiable loss function.

This implementation is ideal for researchers and students looking to understand the inner workings of second-order optimization methods. It is based on the original paper: [Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training](https://arxiv.org/abs/2305.14342) by Liu, et al.

## Key Features

  * **From-Scratch Implementation**: Clear, commented code designed for learning and understanding.
  * **Unified Optimizer Class**: A single `Sophia` class handles both `'gnb'` and `'hutchinson'` estimators.
  * **Vectorized & Robust**: The Hutchinson estimator is implemented in a vectorized way that is both efficient and robust against common PyTorch autograd pitfalls.
  * **Minimal Dependencies**: Requires only PyTorch.
  * **Working Examples**: Includes complete, runnable examples for both a classification and a regression task to demonstrate proper usage.

## Installation

This is not a package. To use the optimizer, simply copy the `Sophia` class from the `sophia_optimizer.py` file below into your project.

## The Optimizer Code

Here is the complete, final implementation of the optimizer.

`sophia_optimizer.py`

```python
import torch
import torch.nn as nn
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
```

## Usage

Below are two complete examples demonstrating how to use the `Sophia` optimizer for different tasks.

### Example 1: Classification (Sophia-G)

For standard classification tasks with `CrossEntropyLoss`, we use the default `'gnb'` estimator.

> **Important Usage Pattern:**
>
> 1.  Call `loss.backward(retain_graph=True)` to get the main gradients while keeping the computation graph alive for the Hessian estimation.
> 2.  Pass the model's raw `logits` to the `optimizer.step()` function.

```python
import torch
import torch.nn as nn
from sophia_optimizer import Sophia

# --- Setup Model and Data ---
class ClassifierNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

INPUT_DIM, HIDDEN_DIM, NUM_CLASSES, BATCH_SIZE = 20, 64, 5, 128
X_class = torch.randn(BATCH_SIZE, INPUT_DIM)
y_class = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))

# --- Initialize ---
model = ClassifierNN(INPUT_DIM, HIDDEN_DIM, NUM_CLASSES)
optimizer = Sophia(model.parameters(), lr=1e-3, rho=0.03) # Defaults to hessian_computation_type='gnb'
criterion = nn.CrossEntropyLoss()

# --- Training Loop ---
print("Training classifier with Sophia-G...")
for epoch in range(1001):
    optimizer.zero_grad()
    logits = model(X_class)
    loss = criterion(logits, y_class)
    
    # Must retain graph for the GNB estimator's second backward pass
    loss.backward(retain_graph=True)
    
    optimizer.step(logits=logits)
    
    if epoch % 200 == 0:
        print(f"Epoch {epoch:04d} | Loss: {loss.item():.4f}")
```

### Example 2: Regression (Sophia-H)

For regression or any task with a custom loss function, we use the `'hutchinson'` estimator.

> **Important Usage Pattern:**
>
> 1.  Set `hessian_computation_type='hutchinson'` when creating the optimizer.
> 2.  Define a `closure` function that calculates and returns the loss. **Do not call `.backward()` inside the closure.**
> 3.  Pass this `closure` to the `optimizer.step()` function.

```python
import torch
import torch.nn as nn
from sophia_optimizer import Sophia

# --- Setup Model and Data ---
class RegressionNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

INPUT_DIM, HIDDEN_DIM, BATCH_SIZE = 20, 64, 128
X_reg = torch.randn(BATCH_SIZE, INPUT_DIM)
y_reg = torch.randn(BATCH_SIZE, 1) 

# --- Initialize ---
model = RegressionNN(INPUT_DIM, HIDDEN_DIM)
optimizer = Sophia(
    model.parameters(), 
    lr=1e-4, 
    rho=0.05,
    hessian_computation_type='hutchinson' # Set explicitly
)
criterion = nn.MSELoss()

# --- Training Loop ---
print("Training regressor with Sophia-H...")
for epoch in range(1001):
    optimizer.zero_grad()
    
    # The closure only returns the loss.
    def closure():
        outputs = model(X_reg)
        loss = criterion(outputs, y_reg)
        return loss

    current_loss = optimizer.step(closure=closure)
    
    if epoch % 200 == 0:
        print(f"Epoch {epoch:04d} | Loss: {current_loss.item():.4f}")
```

## Hyperparameter Guide

  * `lr` (float): Learning rate. Default: `1e-3`.
  * `betas` (Tuple[float, float]): Coefficients for EMA of momentum (`beta1`) and Hessian (`beta2`). Default: `(0.9, 0.999)`.
  * `rho` (float): The key Sophia hyperparameter. It clips the update `m_t / h_t` to the range `[-rho, rho]`, preventing divergence from noisy Hessian estimates. Default: `0.03`.
  * `weight_decay` (float): Decoupled weight decay coefficient. Default: `0.0`.
  * `k` (int): The frequency of Hessian updates. The Hessian is estimated every `k` steps. Default: `10`.
  * `hessian_computation_type` (str): The method for Hessian estimation. Must be `'gnb'` (for classification) or `'hutchinson'` (for general use). Default: `'gnb'`.
  * `eps` (float): A small constant added to the denominator for numerical stability. Default: `1e-12`.
