import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("Height_Weight_Data.xlsx", "Sheet1")
x = np.array(df.Weight)
y = np.array(df.Height)
x_points = np.linspace(np.floor(min(x).item()), np.ceil(max(x).item()), 1000)
print(np.floor(min(x).item()))
print(np.ceil(max(x).item()))
print(df.dtypes)


class GradientDecent:
    def __init__(self, a, b, c, learning_rate, iterations):
        # quadratic constants / weights
        self.a = a
        self.b = b
        self.c = c
        # learning weight
        self.lr = learning_rate
        # amount of epochs / iterations
        self.iters = iterations
        self.loss = []

    # calculating gradient of weights
    def fit(self, x_data, y_data, epoch_step=1):
        # our predicted quadratic
        y_pred = self.a * (x_data ** 2) + self.b * x_data + self.c
        # loss function
        loss = np.dot(np.ones(len(y_data)), (y_data - y_pred) ** 2)
        if epoch_step != 1:
            print(f"epoch 1: a = {self.a:.3f}, b = {self.b:.3f}, c = {self.c:.3f}, loss = {loss:.8f}")
        for i in range(self.iters):
            # updating predicted quadratic and loss for new weights
            y_pred = self.a * (x_data**2) + self.b * x_data + self.c

            loss = np.dot(np.ones(len(y_data)), (y_data - y_pred) ** 2)
            self.loss.append(loss)

            # print epoch data for each step
            if (i + 1) % epoch_step == 0:
                print(f"epoch {i + 1}: a = {self.a:.3f}, b = {self.b:.3f}, c = {self.c:.3f}, loss = {loss:.8f}")

            # updating weights
            da = np.dot(x_data ** 2, y_data - y_pred)
            db = np.dot(x_data, y_data - y_pred)
            dc = np.dot(np.ones(len(y_data)), y_data - y_pred)

            self.a += self.lr * da
            self.b += self.lr * db
            self.c += self.lr * dc

    def predict(self, x_data):
        y_pred = self.a * (x_data**2) + self.b * x_data + self.c
        return y_pred

    def loss(self):
        return self.loss


# starting predictions for the slope and intercept
a_pred = np.random.randn()
b_pred = np.random.randn()
c_pred = np.random.randn()
lr = 0.00000000003
num_iters = 10000000
gradient = GradientDecent(a=a_pred, b=b_pred, c=c_pred, learning_rate=lr, iterations=num_iters)
print(gradient.fit(x, y, epoch_step=10000))

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
