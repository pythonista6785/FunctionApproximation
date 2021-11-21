import sys
import torch
import math
import Utils

def main():
	dtype = torch.float
	device = torch.device("cpu")
	# device = torch.device("cuda:0")   # Uncomment this to run on GPU

	# Create random input and output data
	x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
	y = torch.sin(x)

	# Randomly initialize weights 
	a = torch.randn((), device=device, dtype=dtype)
	b = torch.randn((), device=device, dtype=dtype)
	c = torch.randn((), device=device, dtype=dtype)
	d = torch.randn((), device=device, dtype=dtype)

	learning_rate = 1e-6
	for t in range(2000):
		# Forward pass: compute predicted y
		y_pred = a*x**3 + b*x**2 + c*x + d

		# Compute and print loss
		loss = (y_pred - y).pow(2).sum().item()
		if t % 100 == 99:
			print(t, loss)

		# Backprop to compute gradients of a, b, c, d with respect to loss
		grad_y_pred = 2.0 * (y_pred - y)
		grad_d = grad_y_pred.sum()
		grad_c = (grad_y_pred * x).sum()
		grad_b = (grad_y_pred * x**2).sum()
		grad_c = (grad_y_pred * x**3).sum()

		# Update weights using gradient descent 
		a -= learning_rate * grad_a
		b -= learning_rate * grad_b
		c -= learning_rate * grad_c
		d -= learning_rate * grad_d
	print(f'Result: y = {a.item()}x^3} + {b.item()} x^2 + {c.item()} x + {d.item()} ')

	y_pred = a * x**3 + b * x**2 + c * x + d

	# visualize the results, first convert y, y_pred to numpy
	y_pred = y_pred.detach().numpy()
	y = y.detach().numpy()
	Utils.plot_predicted_vs_actual(y_pred, y)


	if __name__ == "__main__":
		sys.exit(int(main() or 0))