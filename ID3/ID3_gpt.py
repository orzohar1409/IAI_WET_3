import numpy as np

class Branch:
    """
    Represents a decision branch (question) in the tree: whether a given feature value is >= a threshold.
    """
    def __init__(self, feature_index, threshold):
        self.feature_index = feature_index
        self.threshold = threshold
    def ask(self, sample):
        # Returns True if the sample's feature value is >= threshold
        return sample[self.feature_index] >= self.threshold
    def __repr__(self):
        return f"Feature {self.feature_index} >= {self.threshold}"

class Leaf:
    """
    Represents a leaf node in the decision tree, storing the count of each class in that leaf.
    """
    def __init__(self, class_counts):
        # class_counts: dict mapping class label -> count of examples of that class in the leaf
        self.class_counts = class_counts
    def __repr__(self):
        return f"Leaf({self.class_counts})"

class Node:
    """
    Represents an internal node in the decision tree, with a question and two child branches.
    """
    def __init__(self, question, true_branch, false_branch):
        # question: Branch object (feature index and threshold)
        # true_branch: subtree for samples that answer True to the question (feature >= threshold)
        # false_branch: subtree for samples that answer False (feature < threshold)
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch
    def __repr__(self):
        return f"[{self.question}]"

class ID3:
    """
    ID3 decision tree classifier supporting continuous features and dynamic threshold splits.
    """
    def __init__(self, min_for_pruning=1):
        """
        min_for_pruning: minimum number of samples required to continue splitting (pre-pruning).
                         If a node has sample count <= min_for_pruning, it will not be split further.
        """
        self.min_for_pruning = min_for_pruning
        self.root = None

    def fit(self, X, y):
        """
        Build the decision tree classifier from training data.
        X: numpy array or list of shape (n_samples, n_features) with continuous feature values.
        y: numpy array or list of shape (n_samples,) with class labels.
        """
        X = np.array(X)
        y = np.array(y)
        self.root = self._build_tree(X, y)
        return self.root

    def _build_tree(self, X, y):
        # Number of samples and features
        n_samples, n_features = X.shape
        # Count class occurrences at this node
        unique, counts = np.unique(y, return_counts=True)
        class_counts = {int(lbl): int(cnt) for lbl, cnt in zip(unique, counts)}
        # Stopping conditions:
        # 1. All samples have the same class label
        if len(unique) == 1:
            return Leaf(class_counts)
        # 2. Pre-pruning: if number of samples is <= min_for_pruning, stop splitting
        if n_samples <= self.min_for_pruning:
            return Leaf(class_counts)
        # Compute current entropy
        def entropy(labels):
            if len(labels) == 0:
                return 0.0
            _, label_counts = np.unique(labels, return_counts=True)
            probabilities = label_counts / len(labels)
            # Calculate entropy (base-2 logarithm)
            return -float(np.sum([p * np.log2(p) for p in probabilities if p > 0]))
        base_entropy = entropy(y)
        best_gain = 0.0
        best_feature = None
        best_threshold = None
        # Find the best feature and threshold for splitting
        for feature_index in range(n_features):
            feature_values = X[:, feature_index]
            # Sort data by this feature to consider threshold candidates
            sorted_idx = np.argsort(feature_values)
            sorted_X = feature_values[sorted_idx]
            sorted_y = y[sorted_idx]
            # Examine possible split points between distinct values
            for i in range(n_samples - 1):
                if sorted_X[i] != sorted_X[i+1]:
                    threshold = (sorted_X[i] + sorted_X[i+1]) / 2.0
                    # Split data by this threshold
                    left_mask = feature_values < threshold
                    right_mask = feature_values >= threshold
                    left_labels, right_labels = y[left_mask], y[right_mask]
                    if len(left_labels) == 0 or len(right_labels) == 0:
                        continue  # skip splits that produce an empty branch
                    # Compute information gain
                    left_entropy = entropy(left_labels)
                    right_entropy = entropy(right_labels)
                    w_left = len(left_labels) / n_samples
                    w_right = len(right_labels) / n_samples
                    info_gain = base_entropy - (w_left * left_entropy + w_right * right_entropy)
                    # Update best split if gain is higher, or if equal gain and feature index is larger (tie-break)
                    if info_gain > best_gain or (abs(info_gain - best_gain) < 1e-12 and best_feature is not None and feature_index > best_feature):
                        best_gain = info_gain
                        best_feature = feature_index
                        best_threshold = threshold
        # If no split yields positive gain, return a Leaf
        if best_feature is None or best_gain <= 1e-12:
            return Leaf(class_counts)
        # Split the dataset using the best feature and threshold
        left_mask = X[:, best_feature] < best_threshold
        right_mask = X[:, best_feature] >= best_threshold
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]
        # Recursively build subtrees for each branch
        false_branch = self._build_tree(X_left, y_left)    # feature < threshold branch
        true_branch = self._build_tree(X_right, y_right)   # feature >= threshold branch
        # Create a decision node with the best question
        question = Branch(best_feature, best_threshold)
        return Node(question, true_branch=true_branch, false_branch=false_branch)

    def predict(self, X):
        """
        Predict class labels for the samples in X.
        X: numpy array or list of shape (n_samples, n_features).
        Returns: numpy array of predicted class labels.
        """
        X = np.array(X)
        predictions = []
        for sample in X:
            node = self.root
            # Traverse the tree until a leaf is reached
            while isinstance(node, Node):
                if node.question.ask(sample):
                    node = node.true_branch
                else:
                    node = node.false_branch
            # node is a Leaf
            if isinstance(node, Leaf):
                counts = node.class_counts
                if counts:
                    # Choose class with highest count; if tie, choose the class with the larger label value
                    max_count = max(counts.values())
                    best_classes = [cls for cls, cnt in counts.items() if cnt == max_count]
                    prediction = max(best_classes)
                else:
                    prediction = None
            else:
                prediction = None
            predictions.append(prediction)
        return np.array(predictions)
