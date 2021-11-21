from numpy import *
import matplotlib.pyplot as plt

def plot_predicted_vs_actual(ypred, yactual):
    mean_abs_error = sum(abs(ypred-yactual)) / len(ypred)
    step_size = 20    #plot every 2th point 
    a_pred = [ypred[i] for i in range(0, len(ypred)) if i%step_size==0]
    b_actual = [yactual[i] for i in range(0, len(ypred)) if i%step_size==0]
    t = linspace(0, len(a_pred), len(a_pred))
    plt.plot(t, a_pred, 'red', linestyle='dashed', label='predicted')
    plt.plot(t, b_actual, 'blue', label='actual')
    plt.scatter(t, a_pred, marker='o', s=10, color='red', label='predicted')
    plt.scatter(t, b_actual, marker='o', s=10, color='blue', label='actual')
    plt.legend()
    plt.title('mean absolute error = '+ str(mean_abs_error))
    plt.show()
