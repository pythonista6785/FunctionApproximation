import sys
import numpy as np
import math 
import Utils 


def main():
    '''
    y = sinx 
    approximate with:
    y = a*x**3 + b*x**2 + c*x + d 
    The first version just uses Numpy to determine
    the a,b,c and d values. Note that we are computing
    gradients of the loss with respect to each varible 
    ourselves. For example the gradient of 'y_pred' with 
    respect to 'a' is 'x**3' 
    '''
    # create random input and output data
    x = np.linspace(-math.pi, math.pi, 2000)
    y = np.sin(x)

    #randomly initialize weights 
    a = np.random.randn()
    b = np.random.randn()
    c = np.random.randn()
    d = np.random.randn()

    learning_rate = 1e-6
    for t in range(2000):
        # forward pass: compute predcited y
        # y = a x^3 + b x^2 + c x + d
        y_pred = a*x**3 + b*x**2+c*x+d

        # Compute and print loss
        loss = np.square(y_pred -y).sum()
        if t % 100 == 99:
            print(t, loss)

        # Backprop to compute gradients of a,b,c and d with respect to loss
        grad_y_pred = 2.0 * (y_pred - y)
        grad_d = grad_y_pred.sum()
        grad_c = (grad_y_pred * x).sum()
        grad_b = (grad_y_pred * x**2).sum()
        grad_a = (grad_y_pred * x**3).sum()

        #update weights 
        a -= learning_rate * grad_a
        b -= learning_rate * grad_b
        c -= learning_rate * grad_c
        d -= learning_rate * grad_d

    print(f'Result: y = {a} x^3 + {b} x^2 + {c} x + {d}')
    y_pred = a*x**3 + b*x**2 + c*x + d
    Utils.plot_predicted_vs_actual(y_pred, y)


if __name__ == "__main__":
    sys.exit(int(main() or 0))