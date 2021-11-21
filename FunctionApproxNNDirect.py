import sys
import torch
import math
import Utils 


def main():
    # Create Tensors to hold input and output. 
    x = torch.linspace(-math.pi, math.pi, 2000)
    y = torch.sin(x)

    xx = torch.reshape(len(x), 1)

    # Use the nn package to define our model as sequence of layers. nn.Sequential 
    # is a Module which contains other Modules, and applies them in sequence to 
    # produce its output. The Linear Module computes output from input using a
    # linear function, and holds internal Tensors for its weight and bias. 
    # The Flatten layer flatens the output of the lineat to a 1D tensor, 
    # to mathc the shape of y.

    model = torch.nn.Sequentail(
        torch.nn.Linear(1, 5),
        torch.nn.Tanh(),
        torch.nn.Linear(5, 1),
        torch.nn.Tanh(),
        torch.nn.Flatten(0, 1)
        )

    # The nn package also contains definitions of popular loss functions; in this
    # case we will use Mean Squared Error (MSE) as our loss function.
    loss_fn = torch.nn.MSELoss(reduction='sum')

    learning_rate = 0.0001 #1e-6
    for t in range(20000):

        # Forward pass: compute predicted y by passing x to the model. Module objects
        # override the __call__ operator so you can call them like functions. When 
        # doing so you pass a Tensor of input data to the Module and it produces 
        # a Tensor of output data.
        y_pred = model(xx)

        # Compute and print loss. We pass Tensors containing the predicted and true 
        # values of y, and the loss function returns a Tensor containing the loss.
        loss = loss_fn(y_pred, y)
        if t % 100 == 99:
            print(t, loss.item())

        # Zero of the gradients before running the backward pass.
        model.zero_grad()

        # Backward: compute gradient of the loss with respect to all the learnables 
        # parameters with requires_grad=True, so this call will compute gradients for 
        # all learnable parameters in the model
        loss.backward()

        # Update the weights using gradient descent. Each parameter is a Tensor, so
        # we can access its gradients like we did before 
        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad

    # You can access the first layer of model as model[0]
    linear_layer = model[0]

    y_pred = model(xx)
    y_pred = y_pred.detach().numpy()
    y = y.detach().numpy()
    Utils.plot_predicted_vs_actual(y_pred, y)

if __name__ == "__main__":
    sys.exit(int(main() or 0))
