from SREDT.SymbolicClassifier import SymbolicClassifier
from SREDT.utils import gini, splitSetOnFunction, readable_deap_function
from numpy import ndarray, array, stack, bincount, max, empty, log2, ceil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from threading import Lock
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
        random_state (int): The random seed for reproducibility.
        nb_processes (int): The maximum number of processes to use for parallelization.
        A maximum of 2**ceil(log2(nb_processes) + 1) + max_depth - ceil(log2(nb_processes)) threads besides the processes will be run.
    Methods:
        fit(X, y): Fit the classifier to the training data.
        predict(X): Predict the class labels for the input features.
    """
    def __init__(self, function_set=set(('add', 'mul')), generations=1000, population_size=100, max_depth=6, algorithm='eaSimple', max_expression_height=3, cost_complexity_threshold=None, random_state=41, nb_processes=8, verbose=2):
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
        
        self.verbose = verbose
        self.random_state = random_state
        self.generations = generations
        self.population_size = population_size
        self.max_depth = max_depth
        self.root = None
        self.algorithm = algorithm
        self.max_expression_height = max_expression_height
        self.cost_complexity_threshold = cost_complexity_threshold
        self.nb_processes = nb_processes
        self.parallelization_height = ceil(log2(nb_processes))
        self.initial_parallelization_height = self.parallelization_height

    def fit(self, X, y):
        self.nb_classes = max(y) + 1 if isinstance(y, ndarray) else max(y.values) + 1

        # we create a toolbox in the SREDTClassifier in order to be able to compile the function outside of the classifier
        toolbox = SymbolicClassifier.make_toolbox(self.function_set, self.max_expression_height, X.shape[1], arithmetic=self.arithmetic)
        
        
        # Initialize the executor for parallel processing if parallelization is enabled
        if self.parallelization_height > 0:
            global executor
            if executor is None:
                # the max number of processes running at the same time is the number of leaves in the tree of parallelization_height
                executor = ProcessPoolExecutor(max_workers=self.nb_processes)
            # the max number of threads running at the same time is the number of nodes in the tree of parallelization_height
            # plus max_depth - parallelization_height as parallelization_height is increased as leaves are made
            # so we need to account for nodes from the top of the tree to the root of the parallelized subtree
            thread_executor = ThreadPoolExecutor(max_workers=2**(self.parallelization_height + 1) - 1 + self.max_depth - self.parallelization_height)
            height_lock = Lock()
            
        SR_params = {
                'random_state': self.random_state,
                'nb_classes': self.nb_classes,
                'function_set': self.function_set,
                'max_expression_height': self.max_expression_height,
                'arithmetic': self.arithmetic,
                'verbose': self.verbose,
            }
        
        def build_SREDT(X, y, depth=0):
            def make_leaf():
                class_distribution = bincount(y, minlength=self.nb_classes)
                majority_class = class_distribution.argmax()
                self.print_if_verbose(1, f"Leaf at depth {depth} with majority class: {majority_class} at predominance: {class_distribution[majority_class]/len(y)}")
                if self.parallelization_height > 0:
                    with height_lock:
                        # leaves liberate resources so parallelization height can be increased
                        # in a way to keep the number of processes running at the same time constant
                        # being equivalent to making the parallelized subtree start on deeper levels
                        self.parallelization_height += 2**(int(self.parallelization_height - self.initial_parallelization_height) - depth + 1)
                return SREDT_leaf(majority_class=majority_class)
            
            # make a leaf if there are no samples left or if the maximum depth is reached
            y_gini = gini(y, nb_classes=self.nb_classes)
            if y_gini < 0.1 or depth >= self.max_depth:
                return make_leaf()

            # evaluating the symbolic regression classifier is done in a separate process
            # as it is the most expensive operation that benefits from being parallelized
            if depth > 0 and self.parallelization_height > 0:
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
                return make_leaf()
            
            
            if self.parallelization_height > 0:
                with height_lock:
                    current_parallelization_height = self.parallelization_height
                self.print_if_verbose(2, f"Current parallelization height: {current_parallelization_height}")
            else:
                current_parallelization_height = 0
            
            # if the current depth is less than the parallelization height, run the left and right branches in parallel
            # this allows to parallelize the training of the SR classifiers
            if depth <= current_parallelization_height - 1:
                left_future = thread_executor.submit(build_SREDT, left, left_labels, depth + 1)
                right_future = thread_executor.submit(build_SREDT, right, right_labels, depth + 1)
                left = left_future.result()
                right = right_future.result()
                return SREDT_node(best_function, threshold, best, left, right, self.arithmetic)
            else:
                return SREDT_node(best_function, threshold, best, build_SREDT(left, left_labels, depth + 1), build_SREDT(right, right_labels, depth + 1), self.arithmetic)
            
        self.root = build_SREDT(X, y)
        
        if self.parallelization_height > 0:
            thread_executor.shutdown(wait=True)
            executor.shutdown(wait=True)
    
    def print_if_verbose(self, verbose_level, *args, **kwargs):
        if self.verbose >= verbose_level:
            print(*args, **kwargs)

    def predict(self,X):
        return self.root(*X.T)

    def __str__(self):
        return self.root.__str__()