import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate 10 points of data x and y
x = np.random.rand(10, 1)  # 10 x points
y = 2 * x + np.random.randn(10, 1)  # 10 y points

theta0_eg = 0
theta1_eg = 0 

def h(x1, theta0, theta1, x0=1):
    return (theta0 * x0) + (theta1 * x1)

def J_func(theta0, theta1, m, x_matrix, y_matrix):
    mult = 1 / (2 * m)
    
    # Calculate hypothesis function values
    h_theta_ret = h(x_matrix, theta0, theta1)
    
    # Calculate squared error
    squared_error = np.sum((h_theta_ret - y_matrix) ** 2)
    
    # return cost
    return mult * squared_error

def Grad_desc(theta0, theta1, alpha, tol, max_iter, x_matrix, y_matrix):
    m = len(x_matrix)
    table_data = []
    cost_history = []
    cost_history.append(J_func(theta0, theta1, m, x_matrix, y_matrix))
    
    for iterations in range(max_iter):
        h_theta_ret = h(x_matrix, theta0, theta1)

        #update thetas
        theta0 = theta0 - (alpha / m) * np.sum(h_theta_ret - y_matrix) # remember x0 = 1!
        theta1 = theta1 - (alpha / m) * np.sum((h_theta_ret - y_matrix) * x_matrix) 
        
        cost_new = J_func(theta0, theta1, m, x_matrix, y_matrix)

        #store cost history for error calc
        cost_history.append(cost_new)
        
        table_data.append([iterations, cost_new, theta0, theta1])
        
        #reverse indexing to avoid looping
        if iterations > 0 and abs(cost_history[-1] - cost_history[-2]) < tol: #-1 is first index from the beack of the array (behind)
            break
    
    # Plot cost func vs thetas in 3-D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Create grid with 100 values between -2 and 2
    theta0_vals = np.linspace(-2, 2, 100) 
    theta1_vals = np.linspace(-2, 2, 100)
    theta0_mesh, theta1_mesh = np.meshgrid(theta0_vals, theta1_vals) #make  a mesh of points (cartesian plane)
    J_vals = np.zeros_like(theta0_mesh)
    
    #find costs at specific points
    for i in range(len(theta0_vals)):
        for j in range(len(theta1_vals)):
            J_vals[i, j] = J_func(theta0_vals[i], theta1_vals[j], m, x_matrix, y_matrix)
    
    ax.plot_surface(theta0_mesh, theta1_mesh, J_vals, cmap='viridis')
    
    ax.set_xlabel('Theta0')
    ax.set_ylabel('Theta1')
    ax.set_zlabel('Cost Function')
    ax.set_title('Cost Function Surface')
    
    # Plot starting point (initial theta0 and theta1) in red
    ax.scatter(theta0_eg, theta1_eg, J_func(theta0_eg, theta1_eg, m, x_matrix, y_matrix), color='red', marker='o', label='Start point')
    
    # Plot optimal point (optimal theta0 and theta1) in blue -> global only
    ax.scatter(theta0, theta1, J_func(theta0, theta1, m, x_matrix, y_matrix), color='blue', marker='o', label='Optimal point')

    plt.legend()
    plt.show()
    
    return np.array([theta0, theta1]), table_data

# driver code 
alpha = 0.01
tolerance = 1e-6 #can be replaced through normal equ
max_iterations = 1000

theta0_optimal, table_data = Grad_desc(theta0_eg, theta1_eg, alpha, tolerance, max_iterations, x, y)
print("Optimal theta0:", theta0_optimal)
print("Table Data:")
for row in table_data:
    print(row)

# Scatter plot of the data points with scatter plot
plt.scatter(x, y, color='blue', label='Data points')


x_line = np.linspace(0, 1, 100)[:, np.newaxis]  # Generating x values for the line
y_line = h(x_line, theta0_optimal[0], theta0_optimal[1])  # Computing y values using the optimal theta
plt.plot(x_line, y_line, color='red', label='Regression line')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression')
plt.legend()
plt.grid(True)
plt.show()
