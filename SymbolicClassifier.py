import operator
import random
from numpy import ndarray, array, searchsorted, argsort, isfinite, logical_not
from deap import base, creator, gp, tools, algorithms
from pandas import DataFrame
from scipy.optimize import minimize_scalar
from SREDT.utils import split_gini
from warnings import warn

def make_toolbox(function_set, max_expression_height, nb_args, arithmetic=True):
        """
        Create a DEAP toolbox with the given function set and maximum expression height.
        """
        if arithmetic:
            pset = gp.PrimitiveSetTyped("main", [float for _ in range(nb_args)], float)
            if 'add' in function_set:
                pset.addPrimitive(operator.add, [float, float], float)
            if 'mul' in function_set:
                pset.addPrimitive(operator.mul, [float, float], float)
            if 'square' in function_set:
                pset.addPrimitive(SymbolicClassifier.square, [float], float)
            if 'sub' in function_set:
                pset.addPrimitive(operator.sub, [float, float], float)
            if 'div' in function_set:
                pset.addPrimitive(SymbolicClassifier.div, [float, float], float)
            if 'sqrt' in function_set:
                pset.addPrimitive(SymbolicClassifier.sqrt, [float], float)

        else:
            pset = gp.PrimitiveSetTyped("main", [bool for _ in range(nb_args)], bool)
            if 'and' in function_set:
                pset.addPrimitive(operator.and_, [bool, bool], bool)
            if 'or' in function_set:
                pset.addPrimitive(operator.or_, [bool, bool], bool)
            if 'not' in function_set:
                pset.addPrimitive(logical_not, [bool], bool)
            if 'xor' in function_set:
                pset.addPrimitive(operator.xor, [bool, bool], bool)

        if "FitnessMin" not in creator.__dict__:
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if "Individual" not in creator.__dict__:
            creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

        # Create the toolbox and register the genetic programming operations
        toolbox = base.Toolbox()
        toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=min(max_expression_height, 3))
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=pset)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
        toolbox.pset = pset
        
        # Limit the expression length by limiting the height of the tree
        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_expression_height))
        toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_expression_height))
        return toolbox

class SymbolicClassifier:
    """
    A classifier using genetic programming to evolve a symbolic expression
    with a threshold to be used as a decision boundary for a decision tree.
    Arguments:
        X (np.ndarray or pd.DataFrame): The input features (the classifier needs the dataset at creation to properly set up genetic programming functions).
        y (np.ndarray or pd.DataFrame): The target labels.
    Parameters:
        function_set (set): The set of functions to be used in the symbolic expression (arithmetic and logical expressions cannot be mixed).
        max_expression_height (int): The maximum height of the tree representing a symbolic expression.
        algorithm (str): The genetic programming algorithm to use (either 'eaSimple' or 'eaMuPlusLambda').
        arithmetic (bool): Whether to use arithmetic functions (if True, it is assumed arithmetic functions are used so logical ones cannot be used).
        random_state (int): The random seed for reproducibility.
        nb_classes (int): The number of classes in the target labels (if None, it is inferred from the labels).
    Methods:
        fit(generations=100, population_size=200): Fit the symbolic classifier to the data.
        evalSplit(individual): Evaluate the fitness of an individual by calculating the Gini impurity of the split it represents.
        findBestThreshold(function): Find the threshold that minimizes the Gini impurity of the split represented by the given function.
    """

    def __init__(self, X, y, function_set=set(('add', 'mul')), algorithm='eaSimple', max_expression_height=3, arithmetic=True, random_state=41, nb_classes=None, toolbox=None, verbose=2):
        if not isinstance(X, ndarray):
            if isinstance(X, DataFrame):
                self.X = X.values
            else:
                raise ValueError("X must be a numpy array or a pandas DataFrame.")
        else:
            self.X = X
        if not isinstance(y, ndarray):
            if isinstance(y, DataFrame):
                self.y = y.values
            else:
                raise ValueError("y must be a numpy array or a pandas DataFrame.")
        else:
            self.y = y
        if len(self.X) != len(self.y):
            raise ValueError("X and y must have the same number of samples.")

        self.nb_classes = nb_classes if nb_classes is not None else max(self.y) + 1 if isinstance(self.y, ndarray) else max(self.y.values) + 1

        if not isinstance(function_set, set):
            try:
                self.function_set = set(function_set)
            except TypeError:
                raise ValueError("function_set must be a set or iterable.")
        else:
            self.function_set = function_set

        # the gini cache is used to store the Gini impurity and best threshold for already evaluated individuals inside evalSplit
        self.gini_cache = dict()
        # the gini_split_cache is used to store already threshold splits for a given individual inside findBestThreshold
        self.gini_split_cache = dict()
        self.gini_cache_count = 0
        self.gini_split_cache_count = 0

        self.verbose = verbose

        self.random_state = random_state
        self.algorithm = algorithm
        self.max_expression_height = max_expression_height
        self.arithmetic = arithmetic
        if toolbox is not None:
            self.toolbox = toolbox
        else:
            self.toolbox = make_toolbox(function_set, max_expression_height, self.X.shape[1], arithmetic=arithmetic)
        self.toolbox.register("evaluate", self.evalSplit)
        
    
    @staticmethod
    def square(x):
        return operator.pow(x, 2)
    
    @staticmethod
    def sqrt(x):
        if isinstance(x, ndarray):
            return array([float('nan') if xi < 0 else operator.pow(xi, 0.5) for xi in x])
        if x < 0:
            return float('nan')
        return operator.pow(x, 0.5)
    
    @staticmethod
    def div(x, y):
        if isinstance(x, ndarray) and isinstance(y, ndarray):
            return array([float('inf') if yi == 0 and xi > 0 else float('-inf') if yi == 0 and xi < 0 else operator.truediv(xi, yi) for xi, yi in zip(x, y)])
        # if y == 0 and x > 0:
        #     return float('inf')
        # elif y == 0 and x < 0:
        #     return float('-inf')
        if y == 0:
            return 0
        return operator.truediv(x, y)
    
    def print_if_verbose(self, verbose_level, *args, **kwargs):
        if self.verbose >= verbose_level:
            print(*args, **kwargs)

    def evalSplit(self, individual):
        computedHash = str(individual)
        cached_result = self.gini_cache.get(computedHash)
        if cached_result is not None:
            if self.verbose >= 2:
                self.gini_cache_count += 1
            if self.arithmetic:
                return (cached_result[1],)
            return cached_result,
        func = self.toolbox.compile(expr=individual)
        if not self.arithmetic:
            try:
                left_mask = func(*self.X.T).astype(bool)
            except Exception:
                warn("Vectorized evaluation failed in evalSplit.")
                left_mask = array([func(*features) for features in self.X], dtype=bool)
            right_mask = ~left_mask
            gini_results = split_gini(self.y[left_mask], self.y[right_mask], nb_classes=self.nb_classes)
            self.gini_cache[computedHash] = gini_results
            return gini_results,
        _, best_gini = self.findBestThreshold(func, individual, computedHash=computedHash)
        return (best_gini,)


    def findBestThreshold(self, function, individual, computedHash=None):
        """
        Find the threshold that best splits the dataset based on the predictions of the given function.
        Returns:
            The best threshold value.
        """
        if len(self.X) == 1:
            return function(*self.X[0]), 0.
        # First evaluate the function on the dataset
        try:
            predictions = function(*self.X.T)
        except Exception:
            warn("Vectorized evaluation failed in findBestThreshold.")
            predictions = array([function(*features) for features in self.X])
        # Sort the predictions and the corresponding labels
        prediction_order = argsort(predictions)
        predictions = predictions[prediction_order]
        sorted_labels = self.y[prediction_order]
        self.gini_split_cache.clear()
        def gini_split_on(threshold_i):
            """
            Calculate the Gini impurity of the split at the given threshold index.
            """
            if not isinstance(threshold_i, int):
                threshold_i = int(threshold_i)
            result = self.gini_split_cache.get(threshold_i)
            if result is not None:
                if self.verbose >= 2:
                    self.gini_split_cache_count += 1
                return result
            result = split_gini(sorted_labels[:threshold_i+1], sorted_labels[threshold_i+1:], nb_classes=self.nb_classes)
            self.gini_split_cache[threshold_i] = result
            return result

        def gini_from_value(threshold):
            """
            Calculate the Gini impurity of the split at the given threshold value.
            """
            return gini_split_on(searchsorted(predictions, threshold)-1)
        
        # Filter out non-finite predictions for threshold finding
        finite_predictions = predictions[isfinite(predictions)]
        if len(finite_predictions) == 0:
            return 0.0, gini_from_value(0.0)
        
        result = minimize_scalar(gini_from_value, bounds=(finite_predictions[0],finite_predictions[-1]))
        if result.success:
            result_gini = gini_from_value(result.x)
            if computedHash is None:
                computedHash = str(individual)
            self.gini_cache[computedHash] = (result.x, result_gini)
            return result.x, result_gini
        else:
            warn(f"Threshold finding failed: {result.message}. Returning 0.0 as threshold.")
            return 0.0, gini_from_value(0.0)
        
    def fit(self, generations=1000, population_size=100):
        random.seed(self.random_state)
        pop = self.toolbox.population(n=population_size)
        
        if self.algorithm == 'eaSimple':
            algorithms.eaSimple(pop, self.toolbox, cxpb=0.5, mutpb=0.2, ngen=generations, verbose=False)
        else:
            algorithms.eaMuPlusLambda(pop, self.toolbox, mu=population_size, lambda_=population_size, cxpb=0.5, mutpb=0.3, ngen=generations, verbose=False)

        # Save the best individual, its function and its optimal threshold
        if self.arithmetic:
            self.print_if_verbose(2, f"Cache hits: {self.gini_cache_count}, Threshold search cache hits: {self.gini_split_cache_count}")
        else:
            self.print_if_verbose(2, f"Cache hits: {self.gini_cache_count}")
        self.best = tools.selBest(pop, 1)[0]
        self.best_function = self.toolbox.compile(expr=self.best)
        if not self.arithmetic:
            self.threshold = None
            self.final_gini = self.evalSplit(self.best)[0]
        else:
            computedHash = str(self.best)
            self.threshold, self.final_gini = self.gini_cache.get(computedHash, (None, None))
            if self.threshold is None:
                self.threshold, self.final_gini = self.findBestThreshold(self.best_function, self.best)
        self.gini_cache.clear()
