import numpy as np
import matplotlib.pyplot as plt

data_x = np.linspace(1.0, 10.0, 100)[:, np.newaxis]
data_y = np.cos(data_x) + 0.1*np.power(data_x,2) + 0.5*np.random.randn(100,1)
data_x /= np.max(data_x)

plt.scatter(data_x, data_y)

w = np.random.randn(2)
alpha = 0.5
tolerance = 1e-5

# Adding bias
data_x = np.hstack((np.ones_like(data_x), data_x))


# Randomly split training data into training and testing portions of size 20
order = np.random.permutation(len(data_x))
portion = 20
test_x = data_x[order[:portion]]
test_y = data_y[order[:portion]]
train_x = data_x[order[portion:]]
train_y = data_y[order[portion:]]

def gradient_descent(w ,x, y):
	y_pred = x.dot(w).flatten()
	error = (y.flatten()- y_pred)
	gradient = -(1/len(x))*error.dot(x)
	return gradient, np.power(error, 2)


# Performing gradient descent
iterations = 1
while True:
	gradient, error = gradient_descent(w, train_x, train_y)
	new_w = w - alpha*gradient

	if np.sum(abs(new_w - w)) < tolerance:
		print("Converged")
		
		plt.plot(data_x, data_x.dot(new_w).flatten())
		plt.show()
		break

	if iterations % 100 == 0:
		print("Iteration = " + str(iterations) + "\n Error = " + str(error))


	iterations += 1
	w = new_w

