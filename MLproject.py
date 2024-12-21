path=r"C:\Users\User\OneDrive\Desktop\electrical.csv"
import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
#from scipy.special import label
from sklearn.cluster import KMeans
#Read the data
data=pd.read_csv(path)
X1 = data.iloc[:,[1,3]].values
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i,init='k-means++', max_iter=300, n_init=10,random_state=0)
    kmeans.fit(X1)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
#Applying K-Means
kmeans = KMeans(n_clusters= 3, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(X1)

#Visiualizing the data
plt.scatter(X1[y_kmeans == 0,0], X1[y_kmeans == 0,1], s=100, c='red', label='Cluster_1')
plt.scatter(X1[y_kmeans == 1,0], X1[y_kmeans == 1,1], s=100, c='green', label='Cluster_2')
plt.scatter(X1[y_kmeans == 2,0], X1[y_kmeans == 2,1], s=100, c='black', label='Cluster_3')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'Yellow', label='Centroids')
plt.title('The final clustered data')
plt.xlabel('Active Power')
plt.ylabel('Voltage')
plt.legend()
plt.show()


from sklearn.model_selection import train_test_split
df = pd.read_csv(path)
x = df.iloc[:, [1,3]]
y=df.iloc[:,-1]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
from sklearn.preprocessing import MinMaxScaler
scale1=MinMaxScaler()
x_train=scale1.fit_transform(x_train)
x_test=scale1.fit_transform(x_test)
from sklearn.neighbors import KNeighborsClassifier
cl1=KNeighborsClassifier(n_neighbors=3,metric='minkowski',p=1)
cl1.fit(x_train,y_train)
y_predict=cl1.predict(x_test)
print(y_predict)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_predict)
print(cm)
from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,y_predict)
print(acc)

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# Extract independent (X) and dependent (y) variables
X = df.iloc[:, 1:2].values  # Position level (assuming it's in column 1, reshape as a 2D array)
y1 = df.iloc[:, 3].values    # Salary (assuming it's in column 3)

# Linear regression model
regressor1 = LinearRegression()
regressor1.fit(X, y1)

# Polynomial regression model
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)  # Transform X to polynomial features
regressor2 = LinearRegression()
regressor2.fit(X_poly, y1)

# Plotting the results
plt.scatter(X, y1, color='red')  # Actual data points
plt.plot(X, regressor1.predict(X), color='blue', label='Linear Regression')  # Linear regression line
plt.plot(X, regressor2.predict(X_poly), color='green', label='Polynomial Regression')  # Polynomial regression curve
plt.title('Linear vs Polynomial Regression')
plt.xlabel("Active Power")
plt.ylabel("Voltage")
plt.legend()
plt.show()
