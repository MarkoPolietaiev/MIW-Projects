import numpy as np


class Node:
    def __init__(self, x, y, score, feature, threshold):
        self.x = x
        self.y = y
        self.score = score
        self.feature = feature
        self.threshold = threshold


class CartDecisionTree:
    tree = {}

    def __init__(self, max_depth, acceptable_impurity):
        self.max_dept = max_depth
        self.acceptable_impurity = acceptable_impurity

    def fit(self, values, classes):
        CartDecisionTree.tree = CartDecisionTree.build_tree(self, values, classes, self.max_dept,
                                                            self.acceptable_impurity)

    def build_tree(self, x, y, length, impurity):
        """
        build tree recursively with help of cart_split function
         - Create a node
         - call recursive build_tree to build the complete tree
        :return:
        """
        x_l, y_l, x_r, y_r, score, feature, threshold = CartDecisionTree.cart_split(x, y,
                                                                                    CartDecisionTree.gini_impurity)
        node = Node(x, y, score, feature, threshold)
        if length == 0 or CartDecisionTree.gini_impurity(node.y) <= impurity:
            return node
        node.left = CartDecisionTree.build_tree(self, x_l, y_l, length - 1, impurity)
        node.right = CartDecisionTree.build_tree(self, x_r, y_r, length - 1, impurity)
        return node

    @staticmethod
    def get_class_for_node(y):
        return np.argmax(np.bincount(y))

    def gini_impurity(y):
        instances = np.bincount(y)
        total = np.sum(instances)
        p = instances / total
        return 1.0 - np.sum(np.power(p, 2))

    @staticmethod
    def split_node(x, y, feature, threshold):
        x_l = []
        y_l = []
        x_r = []
        y_r = []
        for feature_set, classification in zip(x, y):
            if feature_set[feature] > threshold:
                x_r.append(feature_set)
                y_r.append(classification)
            else:
                x_l.append(feature_set)
                y_l.append(classification)
        return np.asarray(x_l), np.asarray(y_l, dtype=np.int64), np.asarray(x_r), np.asarray(y_r, dtype=np.int64)

    @staticmethod
    def get_score_for_split(y, y_l, y_r, impurity_measure):
        left_score = impurity_measure(y_l) * y_l.shape[0] / y.shape[0]
        right_score = impurity_measure(y_r) * y_r.shape[0] / y.shape[0]
        return left_score + right_score

    @staticmethod
    def cart_split(x, y, impurity_measure):
        x_l_best = None
        y_l_best = None
        x_r_best = None
        y_r_best = None
        score_best = None
        feature_best = None
        threshold_best = None
        for feature in range(x.shape[1]):
            start = np.min(x[:, feature])
            end = np.max(x[:, feature])
            for threshold in np.arange(start, end, 10):
                x_l, y_l, x_r, y_r = CartDecisionTree.split_node(x, y, feature, threshold)
                score = CartDecisionTree.get_score_for_split(y, y_l, y_r, impurity_measure)
                if score_best is None or score_best > score:
                    x_l_best = x_l
                    y_l_best = y_l
                    x_r_best = x_r
                    y_r_best = y_r
                    score_best = score
                    feature_best = feature
                    threshold_best = threshold
        return x_l_best, y_l_best, x_r_best, y_r_best, score_best, feature_best, threshold_best

    @staticmethod
    def predict(values):
        current_node = CartDecisionTree.tree
        while True:
            if hasattr(current_node, 'left') and hasattr(current_node, 'right'):
                if values[current_node.feature] > current_node.threshold:
                    current_node = current_node.right
                else:
                    current_node = current_node.left
            else:
                result = CartDecisionTree.get_class_for_node(current_node.y)
                return result
