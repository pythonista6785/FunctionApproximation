import sys
import torch
import math
import Utils


def main():
    '''
    Previously, we computed the gradients ourselves.
    Since PyTorch's AutoGrad can compute the gradients 
    automatically, we will show to do automatic 
    differentiaion in this version othe program.
    '''

    dtype = torch.float
    device = torch.device("cpu")
    # device = torch.device("cuda:0")   #Uncoment this to run on GPU

    # Create Tensors to hold input and outputs
    # By default, requires_grad=False, which indicates that we do not need to 
    # compute gradients with respect to these Tensors during the backward pass.

    x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
    y = torch.sin(x)

    # Create random Tensors for weights. For a third order polynomial, we need 
    # 4 Weights: y = ax^3 + bx^2 + c x + d
    # Settings requires_grad = True indicates that we want to compute gradients with
    # respect to these Tensors during the bakward pass.

    a = torch.randn((), device=device, dtype=dtype, requires_grad=True)
    b = torch.randn((), device=device, dtype=dtype, requires_grad=True)
    c = torch.randn((), device=device, dtype=dtype, requires_grad=True)
    d = torch.randn((), device=device, dtype=dtype, requires_grad=True)

    learning_rate = 1e-6
    for t in range(2000):
        # Forward pass: compute predicted y using operations on Tensors.
        y_pred = a * x**3 + b * x**2 + c *x + d

        # Compute and print loss using operations on Tensors
        # Now loss is a Tensor of shape (1,)
        # loss.item() gets the scalar value held in the loss.
        loss = (y_pred - y).pow(2).sum()
        if t % 100 == 99:
            print(t, loss.item())

        # use autograd to compute the backward pass. This call will compute the 
        # gradient of loss with respect to all Tensors with requires_grad=True.
        # After this call a.grad, b.grad, c.grad and d.grad will be Tensors holding
        # the gradient of the loss with respect to a, b, c, d respectively.
        
        loss.backward()

        # Manually update weights using gradient descent. Wrap in torch.no_grad()
        # because weights have requires_grad=True, but we don't need to track this
        # in autograd.

        with torch.no_grad():
            a -= learning_rate * a.grad
            b -= learning_rate * b.grad
            c -= learning_rate * c.grad
            d -= learning_rate * d.grad

            # Manually zero the gradients after updating weights 
            a.grad = None
            b.grad = None 
            c.grad = None
            d.grad = None
    print(f'Result: y= {a.item()}x^3 + {b.item()}x^2 + {c.item()}x + {d.item()}')
    # visualize the results, first convert y, y_pred to numpy
    y_pred = y_pred.detach().numpy()
    y = y.detach().numpy()
    Utils.plot_predicted_vs_actual(y_pred, y)

if __name__ == "__main__":
    sys.exit(int(main() or 0))