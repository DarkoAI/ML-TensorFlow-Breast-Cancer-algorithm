from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# dataset
dataBC = load_breast_cancer()
dataBC.target[[20, 60, 90]]
list(dataBC.target_names)
dataBC

# split/train/test algorithm
X, y = load_breast_cancer(return_X_y=True)
X_Scaled = MinMaxScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_Scaled, y, test_size=0.4)

# Nearest Neighbors Classifier
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
class_knn = KNeighborsClassifier(n_neighbors=3)
class_knn.fit(X_train, y_train)
print("The score test is: {:.2f}".format(class_knn.score(X_test, y_test)))
