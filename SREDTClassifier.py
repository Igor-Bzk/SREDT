from SREDT.SymbolicClassifier import SymbolicClassifier
from SREDT.utils import gini, splitSetOnFunction, readable_deap_function, split_gini
from numpy import ndarray, array, stack, bincount, max

class SREDT_node:
    """
    A node in the symbolic regression decision tree.
    Attributes:
        function: The symbolic expression representing the split.
        uncompiled_function: The uncompiled version of the symbolic expression for string representation.
        left: The left child node.
        right: The right child node.
    """
    def __init__(self, function, threshold, uncompiled_function, left, right, arithmetic):
        self.function = function
        self.threshold = threshold
        self.uncompiled_function = uncompiled_function
        self.left = left
        self.right = right
        self.arithmetic = arithmetic

    def __call__(self, *args):
        """
        Evaluate the arguments against the symbolic expression and make them continue down the left or right subtree depending on the result.
        """
        rep = self.function(*args)
        if isinstance(rep, ndarray):
            args_arr = stack(args)
            if not self.arithmetic:
            # no threshold for logical functions
                return array([
                    self.left(*args_arr[:, r]) if not rep[r] else self.right(*args_arr[:, r])
                    for r in range(args_arr.shape[1])
                ])
            return array([
                self.left(*args_arr[:, r]) if not rep[r] > self.threshold else self.right(*args_arr[:, r])
                for r in range(args_arr.shape[1])
            ])
        
        # for non-array arguments
        if not self.arithmetic:
            return self.left(*args) if not rep else self.right(*args)
        return self.left(*args) if not rep > self.threshold else self.right(*args)

    def __str__(self, depth=0):
        if self.arithmetic:
            s = f"{depth * '  '}Node: {readable_deap_function(self.uncompiled_function)} > {self.threshold}\n"
        else:
            s = f"{depth * '  '}Node: {readable_deap_function(self.uncompiled_function)}\n"
        s += self.left.__str__(depth + 1)
        s += self.right.__str__(depth + 1)
        return s

class SREDT_leaf:
    """
    A leaf node in the symbolic regression decision tree.
    Attributes:
        majority_class: The majority class label for the leaf node.
    """
    def __init__(self, majority_class):
        self.majority_class = majority_class

    def __call__(self, *args):
        """
        Evaluate the leaf node by returning the majority class label.
        """
        if isinstance(args, ndarray):
            return array([self.majority_class for _ in range(args.shape[0])])
        return self.majority_class
    def __str__(self, depth=0):
        return f"{depth * '  '}Leaf: {self.majority_class}\n"

class SREDTClassifier:
    """
    A symbolic regression decision tree classifier using DEAP to generate splits through genetic programming.
    Parameters:
        function_set (set): A tuple of strings representing the function set to use for genetic programming.
        generations (int): The number of generations to run the genetic programming algorithm.
        population_size (int): The size of the population for the genetic programming algorithm.
        max_depth (int): The maximum depth of the decision tree.
        algorithm (str): The genetic programming algorithm to use ('eaSimple' or 'eaMuPlusLambda').
        max_expression_height (int): The maximum height of the tree representing a symbolic expression.
        A height of n means expressions of at most 2^n - 1 nodes.
        cost_complexity_threshold (float): The threshold for cost complexity pruning (if None, pruning is disabled).
    """
    def __init__(self, function_set=set(('add', 'mul')), generations=1000, population_size=100, max_depth=10, algorithm='eaSimple', max_expression_height=3, cost_complexity_threshold=None, random_state=41):
        if not isinstance(function_set, set):
            try:
                self.function_set = set(function_set)
            except TypeError:
                raise ValueError("function_set must be a set or iterable.")
        else:
            self.function_set = function_set
        
        # Check if the function set contains arithmetic or logical functions
        if self.function_set & {'add', 'mul', 'square', 'sub', 'div', 'sqrt'}:
            if self.function_set & {'and', 'or', 'not', 'xor'}:
                raise ValueError("Arithmetic and logical functions cannot be mixed in the function set.")
            self.arithmetic = True
        else:
            self.arithmetic = False
        
        self.random_state = random_state
        self.generations = generations
        self.population_size = population_size
        self.max_depth = max_depth
        self.root = None
        self.algorithm = algorithm
        self.max_expression_height = max_expression_height
        self.cost_complexity_threshold = cost_complexity_threshold

    def fit(self, X, y):
        self.nb_classes = max(y) + 1 if isinstance(y, ndarray) else max(y.values) + 1
        
        def build_SREDT(X, y, depth=0):
            def make_leaf():
                class_distribution = bincount(y, minlength=self.nb_classes)
                majority_class = class_distribution.argmax()
                print(f"Leaf at depth {depth} with majority class: {majority_class} at predominance: {class_distribution[majority_class]/len(y)}")
                return SREDT_leaf(majority_class=majority_class)
            
            # make a leaf if there are no samples left or if the maximum depth is reached
            if gini(y, nb_classes=self.nb_classes) < 0.1 or depth >= self.max_depth:
                return make_leaf()
            
            SR_params = {
                'function_set': self.function_set,
                'max_expression_height': self.max_expression_height,
                'arithmetic': self.arithmetic,
                'random_state': self.random_state,
                'nb_classes': self.nb_classes
            }
            clf = SymbolicClassifier(X, y, **SR_params)
            clf.fit(generations=self.generations, population_size=self.population_size)

            left, right, left_labels, right_labels = splitSetOnFunction(clf.best_function, X, y, clf.threshold)
            
            print(f"Depth: {depth}, Gini: {gini(y, nb_classes=self.nb_classes)}, Left: {len(left_labels)}, Right: {len(right_labels)}")
            
            # make a leaf if one of the sides is empty or if the split does not improve the Gini impurity significantly
            if len(left_labels) == 0 or len(right_labels) == 0 or (self.cost_complexity_threshold is not None and (gini(y, nb_classes=self.nb_classes) - split_gini(left_labels, right_labels, nb_classes=self.nb_classes) < self.cost_complexity_threshold)):
                return make_leaf()
            
            return SREDT_node(clf.best_function, clf.threshold, clf.best, build_SREDT(left, left_labels, depth + 1), build_SREDT(right, right_labels, depth + 1), self.arithmetic)
            
        self.root = build_SREDT(X, y)

    def predict(self,X):
        return self.root(*X.T)

    def __str__(self):
        return self.root.__str__()