import sys
import torch
import math
import Utils

def main():
    # Create Tesnors to hold input and outputs
    X = torch.linspace(-math.pi, math.pi, 2000)
    y = torch.sin(x)

    # For this example, the output y is a linear function of (x, x^2, x^3), so
    # we can consider it as a linear layer neural network. Let's prepare the 
    # tensor (x, x^2, x^3)

    P = torch.tensor([1,2,3])
    xx = x.unsqueeze(-1).pow(p)

    # In the above code, x.unsqueeze(-1) has shape (2000, 1), and p has shape
    # (3,), for this case, broadcasting semantics will apply to obtain a tensor
    # of shape (2000, 3)

    # Use the nn package to define our model as a sequence of layers. nn.Sequential 
    # is a module which contains other Modules, and applies them in sequence to 
    # produce its outputs. The Linear Module computes output from input using a 
    # linear function, and holds internal Tensors for its weight and bias 
    # The Flatten layer flattens the output of the linear layer to a 1D tensor,
    # match the shape of 'y'

    model = torch.nn.Sequential(
        torch.nn.Linear(3, 1),
        torch.nn.Flatten(0, 1)
        )

    # The nn package also contains definition of popular loss function; in this
    # case we will use Mean Squared Error (MSE) as our loss function.

    Loss_fn = torch.nn.MSELoss(reduction='sum')

    learning_rate = 1e-6
    for t in range(2000):

        # Forward pass: compute predicted y by passing x to the model. Module objects 
        # overide the __call__ operator so you can call them like functions. when 
        # doing so you pass a Tensor of input data to the Module and it produces 
        # a Tensor of output data.

        Y_pred = model(xxx)

        # Compute and print loss. We pass Tensors containing the predicted and true 
        # values of y, and the loss function returns a Tensor containg the loss.

        Loss = loss_fn(y_pred, y)
        if t % 100 == 99:
            print(t, loss.item())

        # Zero the gradients before running the backward pass.
        Model.zero_grad()

        # Backward pass: compute gradient of the loss with respect to all the learnable 
        # parameters of the model. Internally, the parameters of each Module are stored 
        # in Tensors with requires_grad=True, so this call will compute gradients for 
        # all learnable parameters in the model 
        Loss.backward()

        # Update the weights using gradient descent. Each parameter is a Tensor, so
        # we can access its gradients like we did before. 

        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad

        # You can access the first layer of 'model' like accessing the first item of a list
        linear_layer = model[0]

        # For linear layer, its parameters are stored as 'weight' and 'bias'.
        print(f'Result: y = {linear_layer.bias.item()} + {linear_layer.weight[:,0].item()} x + \
               {linear_layer.weight[:, 1].item()} x^2 + {linear_layer.weight[:,2].item()} x^3')
        y_pred = model(xx)
        y_pred = y_pred.detach().numpy()
        y = y.detached().numpy()
        Utils.plot_predicted_vs_actual(y_pred, y)

if __name__ == "__main__":
    sys.exit(int(main() or 0))
