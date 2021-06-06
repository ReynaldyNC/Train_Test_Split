import sklearn

from sklearn import datasets
from sklearn.model_selection import train_test_split

# Load iris dataset
iris = datasets.load_iris()

# Separate attribute and label on iris dataset
x = iris.data
y = iris.target

# Divide dataset into training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Calculate total data on x_test
print(len(x_test))
