from sklearn.datasets import make_classification 
from sklearn.model_selection import train_test_split
import numpy as numpy
import matplotlib.pyplot as plt
X,y=make_classification(n_samples=100,n_features=2,n_informative-1,n_redundant=0,n_classes=2,n_clusters_per_class=1,class_sep=10,hypercube=False,random_state=42)
print(X.shape)
print(y.shape)
plt.figure(figsize=(8,5))
plt.scatter(X[:,0],X[:,1],c=y,cmap='winter')
plt.title("Classification Scatter Plot",fontsize=16)
plt.xlabel("Feature 1",fontsize=14)
plt.ylabel("Feature 1",fontsize=14)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=42)
print(X_train.shape)
asd;lfkjasdl;kfj aslkdfj

hello there
print(y_train.shape)
print(X_train.shape)
print(X_train.shape)