from sklearn.datasets import make_classification 
from sklearn.model_selection import train_test_split
import numpy as numpy
import matplotlib.pyplot as plt
X,y=make_classification(n_samples=100,n_features=2,n_informative=1,n_redundant=0,n_classes=2,n_clusters_per_class=1,class_sep=10,hypercube=False,random_state=42)
print(X.shape)
print(y.shape)
plt.figure(figsize=(8,5))
plt.scatter(X[:,0],X[:,1],c=y,cmap='winter')
plt.title("Classification Scatter Plot",fontsize=16)
plt.xlabel("Feature 1",fontsize=14)
plt.ylabel("Feature 1",fontsize=14)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=42)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
plt.figure(figsize=(8,5))
plt.scatter(X_train[:, 0], X_train[:,1], c=y_train, cmap='winter')
plt.title("Scatter plot for training set", fontsize=16)
plt.xlabel("Feature 1", fontsize=14)
plt.ylabel("Feature 2", fontsize=14)

plt.figure(figsize=(8,5))
plt.scatter(X_test[:,0],X_test[:,1], c=y_test, cmap='winter')
plt.title("Scatter Plot For Test Set", fontsize=16)
plt.xlabel("Feature 1", fontsize=14)
plt.ylabel("Feature 2", fontsize=14)

def step(z):
    return 1 if z > 0 else 0
X = np.insert(X,0,1, axis=1) # insert a column with value 1 in X at first column 
np.ones(X.shape[1])

def perceptron_training(X,y):
    X= np.insert(X,0,1,axis=1) # insert a column with value 1 in X at first co
    weights = np.ones(X.shape[1]) #initialize weights with all 1
    lr=0.001

    for epoch in range(200):
        errors = 0
        for i in range(X.shape[0]):
            weighted_sum = np.dot(weights,X[i])
            y_hat = step(weighted_sum)
            err=y[1] - y_hat
            if err!=0:
                errors +=1
                weights += lr*(err)*X[1]
        print(f"Epoch (epoch+1): Number of misclassifications = {errors}")
    return weights
final_weights = perceptron_training(X_train,y_train)
def plot_decision_boundary(weights, X, y):
    #Extract weights and bias
    bias = weights[0]
    w1 = weights[1]
    w2 = weights[2]
    #Create Linearly spaced x-values
    x_input = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    #Compute corresponding y-values of decision boundary: w1*x1 + w2*x2 + bias = 0
    y_input = -(w1 * x_input + bias) / w2
    #Plot
    plt.figure(figsize=(8, 5))
    plt.plot(x_input, y_input, color='red', linewidth=2, label='Decision Boundary')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='winter')
    plt.title("Perceptron Decision Boundary (Train Data)")
    plt.xlabel("X 1")
    plt.ylabel("X 2")
    plt.legend()
    plt.grid(True)
    plt.ylim(-3,3)
    plt.show()

plot_decision_boundary(final_weights, X_train, y_train)

def perceptron_predict(X, weights):
    X = np.insert(X, 0, 1, axis=1)
    predictions = []
    for x in X:
        z = np.dot(weights, x)
        y_hat = step(z)
        predictions.append(y_hat)
    return np.array(predictions)

y_pred = perceptron_predict(X_test, final_weights)

from sklearn.metrics import accuracy_score

print("Test Accuracy:", accuracy_score(y_test, y_pred))

def perceptron_training(X,y):
    X= np.insert(X,0,1,axis=1) # insert a column with value 1 in X at first column
    weights = np.ones(X.shape[1]) #initialize weights with all 1
    lr=0.001

    for epoch in range(200):
        errors = 0
        for i in range(X.shape[0]):
            weighted_sum = np.dot(weights,X[i])
            y_hat = step(weighted_sum)
            err=y[1] - y_hat
            if err!=0:
                errors +=1
                weights += lr*(err)*X[1]
        print(f"Epoch (epoch+1): Number of misclassifications = {errors}")
    return weights
final_weights = perceptron_training(X_train,y_train)
def plot_decision_boundary(weights, X, y):
    #Extract weights and bias
    bias = weights[0]
    w1 = weights[1]
    w2 = weights[2]
    #Create Linearly spaced x-values
    x_input = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    #Compute corresponding y-values of decision boundary: w1*x1 + w2*x2 + bias = 0
    y_input = -(w1 * x_input + bias) / w2
    #Plot
    plt.figure(figsize=(8, 5))
    plt.plot(x_input, y_input, color='red', linewidth=2, label='Decision Boundary')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='winter')
    plt.title("Perceptron Decision Boundary (Train Data)")
    plt.xlabel("X 1")
    plt.ylabel("X 2")
    plt.legend()
    plt.grid(True)
    plt.ylim(-3,3)
    plt.show()

plot_decision_boundary(final_weights, X_train, y_train)

def perceptron_predict(X, weights):
    X = np.insert(X, 0, 1, axis=1)
    predictions = []
    for x in X:
        z = np.dot(weights, x)
        y_hat = step(z)
        predictions.append(y_hat)
    return np.array(predictions)

y_pred = perceptron_predict(X_test, final_weights)

from sklearn.metrics import accuracy_score

print("Test Accuracy:", accuracy_score(y_test, y_pred))