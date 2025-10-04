from SREDT.SymbolicClassifier import SymbolicClassifier, make_toolbox
from SREDT.utils import gini, splitSetOnFunction, readable_deap_function
from numpy import ndarray, array, stack, bincount, max, empty, log2, ceil, issubdtype, nonzero
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from sklearn.base import BaseEstimator
from threading import Lock
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.preprocessing import LabelEncoder
from warnings import warn
from graphviz import Digraph

executor = None
class SREDT_node:
    """
    A node in the symbolic regression decision tree.
    Attributes:
        function: The symbolic expression representing the split.
        uncompiled_function: The uncompiled version of the symbolic expression for string representation.
        left: The left child node.
        right: The right child node.
    """
    def __init__(self, function, threshold, uncompiled_function, left, right, arithmetic, gini=None, class_distribution=None):
        self.function = function
        self.threshold = threshold
        self.uncompiled_function = uncompiled_function
        self.left = left
        self.right = right
        self.arithmetic = arithmetic
        self.gini = gini
        self.class_distribution = class_distribution

    def __call__(self, *args):
        """
        Evaluate the arguments against the symbolic expression and make them continue down the left or right subtree depending on the result.
        """
        rep = self.function(*args)
        if isinstance(rep, ndarray):
            args_arr = stack(args)
            if not self.arithmetic:
                mask = rep.astype(bool)
            else:
                mask = rep > self.threshold

            # Split the arguments based on the mask from the node's function
            left_args = args_arr[:, ~mask]
            right_args = args_arr[:, mask]

            # Evaluate the left and right functions on the split arguments
            left_results = self.left(*left_args)
            right_results = self.right(*right_args)

            # Reconstruct result array in original order
            result = empty(rep.shape, dtype=left_results.dtype)
            result[~mask] = left_results
            result[mask] = right_results
            return result
        
        # for non-array arguments
        if not self.arithmetic:
            return self.left(*args) if not rep else self.right(*args)
        return self.left(*args) if not rep > self.threshold else self.right(*args)

    def __str__(self, features=None):
        if self.arithmetic:
            return f"Node: {readable_deap_function(self.uncompiled_function, features=features)} > {self.threshold:.4f}\n"
        else:
            return f"Node: {readable_deap_function(self.uncompiled_function, features=features)}\n"

    def subtree_to_string(self, depth=0, features=None):
        """
        Returns a string representation of the subtree rooted at this node.
        """

        s = f"{depth * '  '} {str(self, features=features)}"
        s += self.left.subtree_to_string(depth + 1, features=features) if isinstance(self.left, SREDT_node) else self.left.__str__(depth + 1)
        s += self.right.subtree_to_string(depth + 1, features=features) if isinstance(self.right, SREDT_node) else self.right.__str__(depth + 1)
        return s

    def details(self, labels=None):
        """
        Returns a detailed string representation of the node, including Gini impurity and class distribution.
        """
        s = ""
        if self.gini is not None:
            s += f"Gini: {self.gini:.4f}\n"
        if self.class_distribution is not None:
            s += "Training class distribution:\n"
            for key, value in self.class_distribution.items():
                s += f"Class {key}: {value}\n" if labels is None else f"Class {labels[key]}: {value}\n"
        return s

class SREDT_leaf:
    """
    A leaf node in the symbolic regression decision tree.
    Attributes:
        majority_class: The majority class label for the leaf node.
    """
    def __init__(self, majority_class, gini=None, class_distribution=None):
        self.majority_class = majority_class
        self.gini = gini
        self.class_distribution = class_distribution

    def __call__(self, *args):
        """
        Evaluate the leaf node by returning the majority class label.
        """
        if isinstance(args, ndarray):
            return array([self.majority_class for _ in range(args.shape[0])])
        return self.majority_class
    
    def __str__(self, depth=0, labels=None):
        return f"{depth * '  '}Leaf: Class {self.majority_class}\n" if labels is None else f"{depth * '  '}Leaf: Class {labels[self.majority_class]}\n"

    def details(self, labels=None):
        """
        Returns a detailed string representation of the leaf node, including Gini impurity and class distribution.
        """
        s = ""
        if self.gini is not None:
            s += f"Gini: {self.gini:.4f}\n"
        if self.class_distribution is not None:
            s += "Class distribution during training:\n"
            for key, value in self.class_distribution.items():
                s += f"Class {key}: {value}\n" if labels is None else f"Class {labels[key]}: {value}\n"
        return s

def evalSRClf(X,y, SR_params, generations, population_size):
    """ Evaluate the symbolic regression classifier on the given data.
    Arguments:
        X (ndarray): The input features.
        y (ndarray): The target labels.
        SR_params (dict): Parameters for the symbolic regression classifier.
        generations (int): The number of generations to run the genetic programming algorithm.
        population_size (int): The size of the population for the genetic programming algorithm.
    Returns:
        best (str): The best symbolic expression found (in uncompiled form).
        threshold (float): The threshold value used for splitting.
        final_gini (float): The Gini impurity of the best split.
    """
    clf = SymbolicClassifier(X, y, **SR_params)
    clf.fit(generations=generations, population_size=population_size)
    return clf.best, clf.threshold, clf.final_gini

class SREDTClassifier(BaseEstimator):
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
        random_state (int): The random seed for reproducibility.
        nb_processes (int): The maximum number of processes to use for parallelization.
        A maximum of 2**ceil(log2(nb_processes) + 1) + max_depth - ceil(log2(nb_processes)) threads besides the processes will be run.
    Methods:
        fit(X, y): Fit the classifier to the training data.
        predict(X): Predict the class labels for the input features.
    """
    def __init__(self, function_set=set(('add', 'mul')), generations=1000, population_size=100, max_depth=6, algorithm='eaSimple', max_expression_height=3, cost_complexity_threshold=None, random_state=41, nb_processes=8, verbose=2):
        
        self.function_set = function_set
        self.verbose = verbose
        self.random_state = random_state
        self.generations = generations
        self.population_size = population_size
        self.max_depth = max_depth
        self.algorithm = algorithm
        self.max_expression_height = max_expression_height
        self.cost_complexity_threshold = cost_complexity_threshold
        self.nb_processes = nb_processes
        
    def fit(self, X, y):

        if not isinstance(self.function_set, set):
            try:
                self.function_set = set(self.function_set)
            except TypeError:
                raise ValueError("function_set must be a set or iterable.")
        
        # Check if the function set contains arithmetic or logical functions
        if self.function_set & {'add', 'mul', 'square', 'sub', 'div', 'sqrt'}:
            if self.function_set & {'and', 'or', 'not', 'xor'}:
                raise ValueError("Arithmetic and logical functions cannot be mixed in the function set.")
            self.arithmetic_ = True
        else:
            self.arithmetic_ = False

        X, y = validate_data(self, X, y, reset=True)

        self._estimator_type = "classifier"
        
        # parallelization_depth_ is the number of levels of the tree that get parallelized
        self.parallelization_depth_ = ceil(log2(self.nb_processes))
        self.initial_parallelization_depth_ = self.parallelization_depth_
        
        # we avoid using an LabelEncoder if the labels are already integers
        if not issubdtype(y.dtype, int):
            self.print_if_verbose(2, "Using label encoding.")
            self.label_encoder_ = LabelEncoder()
            y_enc = self.label_encoder_.fit_transform(y)
            self.classes_ = self.label_encoder_.classes_
            self.n_classes_ = len(self.classes_)
        else:
            if hasattr(self, 'label_encoder_'):
                self.label_encoder_ = None
            self.n_classes_ = max(y) + 1 if isinstance(y, ndarray) else max(y.values) + 1
        
        # we create a toolbox in the SREDTClassifier in order to be able to compile the function outside of the classifier
        toolbox = make_toolbox(self.function_set, self.max_expression_height, self.n_features_in_, arithmetic=self.arithmetic_)


        # Initialize the executor for parallel processing if parallelization is enabled
        if self.parallelization_depth_ > 0:
            global executor
            # the max number of processes running at the same time is the number of leaves in the tree of parallelization_depth_
            executor = ProcessPoolExecutor(max_workers=self.nb_processes)
            
            # the max number of threads running at the same time is the number of nodes in the tree of parallelization_depth_
            # plus max_depth - parallelization_depth_ as parallelization_depth_ is increased as leaves are made
            # so we need to account for nodes from the top of the tree to the root of the parallelized subtree
            thread_executor = ThreadPoolExecutor(max_workers=2**(self.parallelization_depth_ + 1) - 1 + self.max_depth - self.parallelization_depth_)
            height_lock = Lock()
            
        SR_params = {
                'random_state': self.random_state,
                'nb_classes': self.n_classes_,
                'function_set': self.function_set,
                'max_expression_height': self.max_expression_height,
                'arithmetic': self.arithmetic_,
                'verbose': self.verbose,
            }

        def build_SREDT(X, y, depth=0):
            def make_leaf(gini=None, class_distribution=None, class_distribution_dict=None):
                if class_distribution is None:
                    class_distribution = bincount(y, minlength=self.n_classes_)
                majority_class = class_distribution.argmax()
                self.print_if_verbose(1, f"Leaf at depth {depth} with majority class: {majority_class} at predominance: {class_distribution[majority_class]/len(y)}")
                if hasattr(self, 'label_encoder_') and self.label_encoder_ is not None:
                    majority_class = self.label_encoder_.inverse_transform([majority_class])[0]
                if self.parallelization_depth_ > 0:
                    if depth < self.parallelization_depth_ - self.initial_parallelization_depth_:
                        warn("Leaf made above parallelization root")
                    else:
                        with height_lock:
                            # leaves liberate resources so parallelization height can be increased
                            # in a way to keep the number of processes running at the same time constant
                            # being equivalent to making the parallelized subtree start on deeper levels
                            self.parallelization_depth_ += 2**(int(self.parallelization_depth_ - self.initial_parallelization_depth_) - depth + 1)
                return SREDT_leaf(majority_class=majority_class, gini=gini, class_distribution=class_distribution_dict)
            
            # make a leaf if there are no samples left or if the maximum depth is reached
            y_gini, class_distribution = gini(y, nb_classes=self.n_classes_, distribution=True)
            non_zero_classes = nonzero(class_distribution)[0]
            if hasattr(self, 'label_encoder_') and self.label_encoder_ is not None:
                class_distribution_dict = dict(zip(self.label_encoder_.inverse_transform(non_zero_classes), class_distribution[non_zero_classes]))
            else:
                class_distribution_dict = dict(zip(non_zero_classes, class_distribution[non_zero_classes]))
            if y_gini < 0.1 or depth >= self.max_depth:
                return make_leaf(gini=y_gini, class_distribution=class_distribution, class_distribution_dict=class_distribution_dict)

            # evaluating the symbolic regression classifier is done in a separate process
            # as it is the most expensive operation that benefits from being parallelized
            if depth > 0 and self.parallelization_depth_ > 0:
                SR = executor.submit(evalSRClf, X, y, SR_params, self.generations, self.population_size)
                best, threshold, final_gini = SR.result()
            else:
                best, threshold, final_gini = evalSRClf(X, y, SR_params, self.generations, self.population_size)

            # the function is compiled outside the classifier as it is a lambda that cannot be pickled and returned by a process
            best_function = toolbox.compile(expr=best)

            left, right, left_labels, right_labels = splitSetOnFunction(best_function, X, y, threshold)

            self.print_if_verbose(1, f"Depth: {depth}, Gini: {y_gini}, Left: {len(left_labels)}, Right: {len(right_labels)}")

            # make a leaf if one of the sides is empty or if the split does not improve the Gini impurity significantly
            if len(left_labels) == 0 or len(right_labels) == 0 or (self.cost_complexity_threshold is not None and (y_gini - final_gini < self.cost_complexity_threshold)):
                return make_leaf(gini=y_gini, class_distribution=class_distribution, class_distribution_dict=class_distribution_dict)

            if self.parallelization_depth_ > 0:
                with height_lock:
                    current_parallelization_depth = self.parallelization_depth_
                self.print_if_verbose(2, f"Current parallelization height: {current_parallelization_depth}")
            else:
                current_parallelization_depth = 0
            
            # if the current depth is less than the parallelization height, run the left and right branches in parallel
            # this allows to parallelize the training of the SR classifiers without getting over the max process number
            if depth <= current_parallelization_depth - 1:
                left_future = thread_executor.submit(build_SREDT, left, left_labels, depth + 1)
                right_future = thread_executor.submit(build_SREDT, right, right_labels, depth + 1)
                left = left_future.result()
                right = right_future.result()
                return SREDT_node(best_function, threshold, best, left, right, self.arithmetic_, gini=y_gini, class_distribution=class_distribution_dict)
            else:
                return SREDT_node(best_function, threshold, best, build_SREDT(left, left_labels, depth + 1), build_SREDT(right, right_labels, depth + 1), self.arithmetic_, gini=y_gini, class_distribution=class_distribution_dict)

        if not issubdtype(y.dtype, int):
            self.root_ = build_SREDT(X, y_enc)
        else:
            self.root_ = build_SREDT(X, y)
        
        if self.parallelization_depth_ > 0:
            thread_executor.shutdown(wait=True)
            executor.shutdown(wait=True)
        return self
    
    def print_if_verbose(self, verbose_level, *args, **kwargs):
        if self.verbose >= verbose_level:
            print(*args, **kwargs)

    def predict(self,X):
        X = validate_data(self, X, reset=False)
        check_is_fitted(self, 'root_')
        return self.root_(*X.T)

    def __str__(self):
        return self.root_.subtree_to_string()

    def display(self, filepath="tree", details=True, features=None, labels=None):
        dot = Digraph("Tree", comment="Decision tree representation", format="png")

        def _add_nodes(dot, node):
            dot.node(str(id(node)), label=f'{node.__str__(labels=labels) if isinstance(node, SREDT_leaf) else node.__str__(features=features)}{"" if not details else node.details(labels=labels)}', shape='box')
            if isinstance(node, SREDT_node):
                dot.edge(str(id(node)), str(id(node.left)), label="False")
                _add_nodes(dot, node.left)
                dot.edge(str(id(node)), str(id(node.right)), label="True")
                _add_nodes(dot, node.right)
        _add_nodes(dot, self.root_)
        dot.render(filepath, format="png", cleanup=True)