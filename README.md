# MINI MLP Build from Scratch (NumPY)

- A small mlp build from using numpy only, this project include the forward propagation and backpropagation written by hand

## Architecture
 
```
Input (2) -> Hidden layer (3 neurons, ReLU) -> Output (1 neuron, Linear)
```
 
- **Hidden layer:** 3 neurons, ReLU activation
- **Output layer:** 1 neuron, linear activation (regression)
- **Loss:** Mean Squared Error (MSE)
- **Optimizer:** Stochastic gradient descent (one weight update per sample)


## Dataset
 
A tiny regression dataset where the target follows `y = x1^2 + 2 * x2`:
 
| x1 | x2 | y  |
|----|----|----|
| 1  | 1  | 3  |
| 2  | 1  | 6  |
| 1  | 2  | 5  |
| 2  | 2  | 8  |
| 3  | 1  | 11 |

- This is just for understanding the backpropagation works underthehood.
### This is the GRAPH LOSS_FUNC w.r.t Per_epoch:
<img width="800" height="600" alt="image" src="https://github.com/user-attachments/assets/1add6403-7d8e-45e0-ba7f-c8745f070c94" />


## Requirements
 
- Python 3.8+
- NumPy
- Matplotlib

```bash
pip install numpy matplotlib
```
 
## Code Overview
 
| Function | Purpose |
|---|---|
| `linear`, `der_linear` | Linear activation and its derivative |
| `relu`, `der_relu` | ReLU activation and its derivative |
| `forward_layer` | Computes pre-activation `z` and activation `h` for one layer |
| `derloss_o_weights` | Gradients of the loss w.r.t. the output layer weights and bias |
| `derloss_h_weights` | Gradients of the loss w.r.t. the hidden layer weights and bias (chain rule) |
| `loss_function` | Computes the MSE over all samples |
| `predict` | Runs inputs through the network with frozen weights (no updates) |
| `mini_mlp` | Training loop; returns the trained weights and the loss history |

## Hyperparameters
 
| Parameter | Default | Notes |
|---|---|---|
| `lr` | `0.01` | If the loss rises or oscillates, lower it |
| `epochs` | `100` | Increase if the loss is still decreasing |


## How It Works
 
For each sample, per epoch:
 
1. **Forward pass:** `z = W·x + b`, `h = relu(z)`, then `y_pred = W_out·h + b_out`
2. **Loss gradient:** `dL/dy_pred = 2 * (y_pred - y)` (from `L = (y - y_pred)^2`)
3. **Backward pass:** the gradient is passed back with the chain rule:
   - output weights: `dL/dW_out = loss_grad * h`
   - hidden activations: `dL/dh = loss_grad * W_out`
   - hidden pre-activations: `dL/dz = dL/dh * relu'(z)`
   - hidden weights: `dL/dW = dL/dz (outer) x`
4. **Update:** `w = w - lr * gradient`
After each epoch, the network is **evaluated** on the dataset with frozen weights, and the MSE is stored in `loss_history`.
 
## Training vs. Evaluation
 
- **Training** updates the weights.
- **Evaluation** (`predict`) only measures predictions with fixed weights. No gradients are computed.
