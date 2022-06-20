from typing import List

class DecisionTreeClassifier:
    
    def __init__(self, max_depth: int):
        
        self.root = None
        self.max_depth = max_depth 
        
    def gini_impurity(self, groups, classes):

        totalsum = 0.0
        for group in groups:
            size = float(len(group))
            if size != 0:
                score = 0.0
                for class_x in classes:
                    pK = [row[-1] for row in group].count(class_x) / size
                    score += pK * pK
                totalsum += score
        gini = 1 - totalsum
        return gini
    
    def split_data(self, X, y):

        best_gini = float("inf")
        for index in range(len(X[0])):
            for x in X:
                left, right = list(), list()
                
                for i,row in enumerate(X):
                    if row[index] <= x[index]:
                        left.append((row, y[i]))
                    else:
                        right.append((row, y[i]))
                groups = [left, right]

                gini_value = self.gini_impurity(groups, list(set(y)))

                if gini_value == 0:
                    return index, x[index], gini_value, groups

                if gini_value < best_gini:
                    best_gini = gini_value
                    best_index = index
                    best_value = x[index]
                    best_groups = groups

        return best_index, best_value, best_gini, best_groups
    
    def build_tree(self, X, y, depth = 0, terminal = False):

        best_index, best_value, best_gini, (left, right) = self.split_data(X, y)
        left_y = [entry[1] for entry in left]
        right_y = [entry[1] for entry in right]
        left = [entry[0] for entry in left]
        right = [entry[0] for entry in right]

        if (best_gini == 0 and len(set(y)) == 1) or depth == self.max_depth:
            node = Node(X, y, best_gini)
            node.flower = max([(flower, y.count(flower)) for flower in set(y)], key=lambda x:x[1])[0]
            return node

        node = Node(X, y, best_gini)
        node.feature_index = best_index
        node.threshold = best_value
        node.flower = max([(flower, y.count(flower)) for flower in set(y)], key=lambda x:x[1])[0]
        node.left = self.build_tree(left, left_y, depth + 1)
        node.right = self.build_tree(right, right_y, depth + 1)
        return node
    
    def fit(self, X: List[List[float]], y: List[int]):
      
        self.root = self.build_tree(X, y)
        return

    def predict(self, X: List[List[float]]):
        
        y = list()
        for x in X:
            y.append(self.predict_case(x))
        return y
    
    def predict_case(self, X):

        node = self.root
        while node.left:
            if X[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.flower


class Node:
    
    def __init__(self, X, y, gini):
        self.X = X
        self.y = y
        self.gini = gini
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None
        self.flower = None