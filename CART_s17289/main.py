from CartDecisionTree import CartDecisionTree
from sklearn.datasets import load_iris


def main():
    iris = load_iris()
    tree = CartDecisionTree(max_depth=2, acceptable_impurity=0.2)
    tree.fit(iris.data, iris.target)
    print("Test data is:", iris.data[4])
    print("Expected class: ", iris.target_names[iris.target[4]])
    prediction = tree.predict(iris.data[4])
    print('Classified as {}'.format(iris.target_names[prediction]))


if __name__ == "__main__":
    main()
