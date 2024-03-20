import numpy as np
import pandas as pd


##TREE NODE

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        self.value = value


class DecisionTreeClassifier:
    def __init__(self, minimum_samples_split=2, max_depth=2):
        self.root = None

        self.minimum_samples_split = minimum_samples_split
        self.max_depth = max_depth

    def __build_tree(self, ds, curr_depth=0):
        x_data, y_data = ds.iloc[:, :-1], ds.iloc[:, -1]
        num_samples, num_features = np.shape(x_data)

        if num_samples >= self.minimum_samples_split and curr_depth <= self.max_depth:

            splitted_data = self.__get_best_split(ds, num_samples, num_features)

            if splitted_data["info_gain"] > 0:
                left_sub_tree = self.__build_tree(splitted_data['dataset_left'], curr_depth + 1)
                right_sub_tree = self.__build_tree(splitted_data['dataset_right'], curr_depth + 1)

                return Node(splitted_data['feature_index'], splitted_data['threshold'], left_sub_tree, right_sub_tree,
                            splitted_data['info_gain'], )

            leaf_value = self.__calculate_leaf_node(y_data)

            return Node(value=leaf_value)

    def __get_best_split(self, ds, num_samples, num_features):

        max_info_gain = -float("inf")
        best_split = {}

        for feature_index in range(num_features):
            feature_values = ds.iloc[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            for threshold in possible_thresholds:
                # get current split
                dataset_left, dataset_right = self.__split(ds, feature_index, threshold)
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    y_data, left_y, right_y = ds.iloc[:, -1], dataset_left.iloc[:, -1], dataset_right.iloc[:, -1]
                    # compute information gain
                    curr_info_gain = self.__information_gain(y_data, left_y, right_y, "gini")
                    # update the best split if needed
                    if curr_info_gain > max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain

        return best_split

    def __split(self, ds, feature_index, threshold):
        left_data = ds.loc[(ds.iloc[:, feature_index] <= threshold)]
        right_data = ds.loc[(ds.iloc[:, feature_index] > threshold)]
        return left_data, right_data

    def __information_gain(self, parent_dataset, left_dataset, right_dataset, mode="entropy"):
        l_weight = len(left_dataset) / len(parent_dataset)
        r_weight = len(right_dataset) / len(parent_dataset)

        gain = self.__gini_index(parent_dataset) - l_weight * self.__gini_index(left_dataset) - r_weight * self.__gini_index(
            right_dataset)
        return gain

    def __gini_index(self, y_data):
        class_labels = np.unique(y_data)
        gini = 0
        for cls in class_labels:
            p_cls = len(y_data[y_data == cls]) / len(y_data)
            gini += p_cls ** 2
        return 1 - gini

    def __calculate_leaf_node(self, Y):
        Y = list(Y)
        return max(Y, key=Y.count)

    def print_tree(self, tree=None, indent=" "):
        if not tree:
            tree = self.root
        if tree.value is not None:
            print(tree.value)
        else:
            print("x_" + str(tree.feature_index), " <= ", tree.threshold, "?", tree.info_gain)
            print("left:")
            self.print_tree(tree.left, indent + indent)
            print("right:")
            self.print_tree(tree.right, indent + indent)

    def fit(self, ds):
        self.root = self.__build_tree(ds)

    def single_predict(self, row):
        return self.make_prediction(row, self.root)

    def predict(self, x_data):

        predictions = [self.make_prediction(row, self.root) for row in x_data.values]
        return predictions

    def make_prediction(self, row, tree):

        if tree is not None and tree.value is not None:
            # print(tree.value)
            return tree.value
        if tree is None:
            return None
        feature_val = row[tree.feature_index]
        # print(str(feature_val)+" --- "+str(tree.feature_index))
        if feature_val <= float(tree.threshold):
            return self.make_prediction(row, tree.left)
        else:
            return self.make_prediction(row, tree.right)

    def accuracy_score(self, y_pred, y_test):
        count = 0
        if len(y_test) != len(y_pred):
            print("not same size")
            return

        for i in range(len(y_test)):
            if y_test[i] == y_pred[i]:
                count += 1

        print(count / len(y_test))
