import datetime

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# date, open, high, low, close, adj close, volume
import csv
import matplotlib.dates as mdates

# Simple Linear Regression on the Swedish Insurance Dataset
from pip._vendor.distlib.compat import raw_input
from sklearn.linear_model import LinearRegression

'''
with open('AAPL3month.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            print(f'\tDate: {row[0]} has close of: {row[4]}.') #has open of: {row[1]}, high of: {row[2]}, low of: {row[3]},
            line_count += 1
    print(f'Processed {line_count} lines.')
    
'''

x = []
y = []
i = 0

with open('AAPL.csv') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x.append(str(row[0]))
        y.append(float(row[1]))
        i = i + 1
'''
plt.plot(x, y, marker='o')
plt.title('AAPL Stock from 3/12/18 - 9/5/18')

plt.xlabel('Date')
plt.ylabel('Close Price')
'''
x_values = [datetime.datetime.strptime(d, "%m/%d/%Y").date() for d in x]

ax = plt.gca()
formatter = mdates.DateFormatter("%Y-%m-%d")
ax.xaxis.set_major_formatter(formatter)
locator = mdates.DayLocator()

ax.xaxis.set_major_locator(locator)

fig, ax = plt.subplots()
ax.plot(x_values, y)

plt.show()

# rotate and align the tick labels so they look better
fig.autofmt_xdate()
print("i is: ", i)
g = []
for x in range(0, i):
    g.append(x)
    # print(x)

w = np.array(g)
z = np.array(y)
plt.plot(w, z, 'o')

m, b = np.polyfit(w, z, 1)

plt.plot(w, m*w + b)

#for j in g:
#    print(j)

plt.show()
'''
a = x[:len(x) // 2]
b = x[len(x) // 2:]

train = y[:len(y) // 2]
test = y[len(y) // 2:]

print("a:\n")

for i in a:
    print(i)
    # print("\n")

print("b:\n")

for i in b:
    print(i)
    # print("\n")

print("training data:\n")

for i in train:
    print(i)
    # print("\n")

print("testing data:\n")

for i in test:
    print(i)
    # print("\n")
'''
# actual weight = 2 and actual bias = 0.9
#g = raw_input("Enter how many months you want to invest for : ")
#type(g)
#print(g)



'''


x = np.linspace(0, 3, 120)
y = 2 * x + 0.9 + np.random.randn(*x.shape) * 0.3

plt.scatter(x,y, label="input data set")
plt.show()


class LinearModel:
    def __call__(self, x):
        return self.Weight * x + self.Bias

    def __init__(self):
        self.Weight = tf.Variable(11.0)
        self.Bias = tf.Variable(12.0)


def loss(y, pred):
    return tf.reduce_mean(tf.square(y - pred))

def train(linear_model, x, y, lr=0.12):
    with tf.GradientTape() as t:
        current_loss = loss(y, linear_model(x))

    lr_weight, lr_bias = t.gradient(current_loss, [linear_model.Weight, linear_model.Bias])
    linear_model.Weight.assign_sub(lr * lr_weight)
    linear_model.Bias.assign_sub(lr * lr_bias)

linear_model = LinearModel()
Weights, Biases = [], []
epochs = 80
for epoch_count in range(epochs):
    Weights.append(linear_model.Weight.numpy())
    Biases.append(linear_model.Bias.numpy())
    real_loss = loss(y, linear_model(x))
    train(linear_model, x, y, lr=0.12)
    #plt.plot(x,y)
    #plt.show()
    #plt.plot(trained x, trained y)
    print(f"Epoch count {epoch_count}: Loss value: {real_loss.numpy()}")

'''

# plt.plot(x,y)
# plt.show()
