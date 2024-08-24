import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read and Distribute Data
df = pd.read_excel("Height_Weight_Data.xlsx", "Sheet1")
x = np.array(df.Weight)
y = np.array(df.Height)
x_points = np.linspace(np.floor(min(x).item()), np.ceil(max(x).item()), 1000)


class GradientDecent:
    def __init__(self, w1, w2, w3, learning_rate, iterations):
        # Weights
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.lr = learning_rate
        self.iters = iterations
        self.loss = []

    # calculating gradient of weights
    def fit(self, x_data, y_data, step=1):
        y_pred = self.w1 * (x_data ** 2) + self.w2 * x_data + self.w3
        
        loss = np.dot(np.ones(len(y_data)), (y_data - y_pred) ** 2)
        
        if step != 1:
            print(f"epoch 1: a = {self.w1:.3f}, b = {self.w2:.3f}, c = {self.w3:.3f}, loss = {loss:.8f}")
        for i in range(self.iters):
            # updating predicted quadratic and loss for new weights
            y_pred = self.w1 * (x_data**2) + self.w2 * x_data + self.w3

            loss = np.dot(np.ones(len(y_data)), (y_data - y_pred) ** 2)
            self.loss.append(loss)

            # Print data for each step
            if (i + 1) % step == 0:
                print(f"epoch {i + 1}: a = {self.w1:.3f}, b = {self.w2:.3f}, c = {self.w3:.3f}, loss = {loss:.8f}")

            # Updating weights
            da = np.dot(x_data ** 2, y_data - y_pred)
            db = np.dot(x_data, y_data - y_pred)
            dc = np.dot(np.ones(len(y_data)), y_data - y_pred)

            self.w1 += self.lr * da
            self.w2 += self.lr * db
            self.w3 += self.lr * dc

    def predict(self, x_data):
        y_pred = self.w1 * (x_data**2) + self.w2 * x_data + self.w3
        return y_pred

    def loss(self):
        return self.loss


# starting predictions for the slope and intercept
a_pred = np.random.randn()
b_pred = np.random.randn()
c_pred = np.random.randn()
lr = 0.00000000003
num_iters = 100000
gradient = GradientDecent(w1=a_pred, w2=b_pred, w3=c_pred, learning_rate=lr, iterations=num_iters)
print(gradient.fit(x, y, step=10000))

# graph line with mat plot lib
y_predicted = gradient.predict(x_points)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8, 6))
m1 = plt.scatter(x, y, color=cmap(0.5), s=10)
# quadratic is correct, this prints out a piecewise line instead ???
# probably because x_data is only defined for certain x values
plt.figure(1)
plt.plot(x_points, y_predicted, color='black', linewidth=2, label="Prediction")
# don't plot with x and loss_func, should plot with epoch instead of x
# doesn't make sense otherwise
plt.figure(2)
plt.yscale('log')
plt.plot(range(num_iters), gradient.loss, color='red', linewidth=2, label="Loss")
plt.show()
