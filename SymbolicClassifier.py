import operator
import random
from numpy import ndarray, array, searchsorted, argsort, isfinite
from deap import base, creator, gp, tools, algorithms
from pandas import DataFrame
from scipy.optimize import minimize_scalar
from SREDT.utils import gini, split_gini

class SymbolicClassifier:
    """
    A classifier using genetic programming to evolve a symbolic expression
    with a threshold to be used as a decision boundary for a decision tree.
    Attributes:
        X (np.ndarray or pd.DataFrame): The input features.
        y (np.ndarray or pd.DataFrame): The target labels.
        function_set (tuple): The set of functions to be used in the symbolic expression (arithmetic and logical expressions cannot be mixed).
        max_expression_height (int): The maximum height of the tree representing a symbolic expression.
        
        algorithm (str): The genetic programming algorithm to use.
        pset (gp.PrimitiveSetTyped): The primitive set for the genetic programming.
        toolbox (base.Toolbox): The DEAP toolbox containing the genetic programming operations.
    Methods:
        fit(generations=100, population_size=200): Fit the symbolic classifier to the data.
        evalSplit(individual): Evaluate the fitness of an individual by calculating the Gini impurity of the split it represents.        
    """

    def __init__(self, X, y, function_set=('add', 'mul'), algorithm='eaSimple', max_expression_height=3, arithmetic=True):
        if not isinstance(X, ndarray):
            if isinstance(X, DataFrame):
                X = X.values
            else:
                raise ValueError("X must be a numpy array or a pandas DataFrame.")
        if not isinstance(y, ndarray):
            if isinstance(y, DataFrame):
                y = y.values
            else:
                raise ValueError("y must be a numpy array or a pandas DataFrame.")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples.")
        self.X = X
        self.y = y
        self.function_set = function_set
        self.algorithm = algorithm
        self.max_expression_height = max_expression_height
        self.arithmetic = arithmetic
        if self.arithmetic:
            self.pset = gp.PrimitiveSetTyped("main", [float for _ in range(X.shape[1])], float)
            if 'add' in function_set:
                self.pset.addPrimitive(operator.add, [float, float], float)
            if 'mul' in function_set:
                self.pset.addPrimitive(operator.mul, [float, float], float)
            if 'square' in function_set:
                self.pset.addPrimitive(self.square, [float], float)
            if 'sub' in function_set:
                self.pset.addPrimitive(operator.sub, [float, float], float)
            if 'div' in function_set:
                self.pset.addPrimitive(self.div, [float, float], float)
            if 'sqrt' in function_set:
                self.pset.addPrimitive(self.sqrt, [float], float)

        else:
            self.pset = gp.PrimitiveSetTyped("main", [bool for _ in range(X.shape[1])], bool)
            if 'and' in function_set:
                self.pset.addPrimitive(operator.and_, [bool, bool], bool)
            if 'or' in function_set:
                self.pset.addPrimitive(operator.or_, [bool, bool], bool)
            if 'not' in function_set:
                self.pset.addPrimitive(operator.not_, [bool], bool)
            if 'xor' in function_set:
                self.pset.addPrimitive(operator.xor, [bool, bool], bool)
                    
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=3)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("compile", gp.compile, pset=self.pset)

        self.toolbox.register("evaluate", self.evalSplit)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=self.pset)

        self.toolbox.decorate("mate", gp.staticLimit(key=len, max_value=17))
        self.toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=17))
        
        # Limit the expression lenght by limiting the height of the tree
        self.toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=self.max_expression_height))
        self.toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=self.max_expression_height))

    @staticmethod
    def square(x):
        return operator.pow(x, 2)
    
    @staticmethod
    def sqrt(x):
        if x < 0:
            return float('nan')
        return operator.pow(x, 0.5)
    
    @staticmethod
    def div(x, y):
        if y == 0 and x > 0:
            return float('inf')
        elif y == 0 and x < 0:
            return float('-inf')
        return operator.truediv(x, y)

    def evalSplit(self, individual):
        func = self.toolbox.compile(expr=individual)
        if not self.arithmetic:
            left_mask = array([func(*features) for features in self.X], dtype=bool)
            right_mask = ~left_mask
            return split_gini(self.y[left_mask], self.y[right_mask]),
        _, best_gini = self.findBestThreshold(func)
        return (best_gini,)

    def findBestThreshold(self, function):
        """
        Find the threshold that best splits the dataset based on the predictions of the given function.
        Returns:
            The best threshold value.
        """
        if len(self.X) == 1:
            return function(*self.X[0]), 0.
        predictions = array([function(*features) for features in self.X])
        prediction_order = argsort(predictions)
        predictions = predictions[prediction_order]
        sorted_labels = self.y[prediction_order]
        
        def gini_split_on(threshold_i):
            if not isinstance(threshold_i, int):
                threshold_i = int(threshold_i)
            return split_gini(sorted_labels[:threshold_i+1], sorted_labels[threshold_i+1:])

        def gini_from_value(threshold):
            return gini_split_on(searchsorted(predictions, threshold)-1)
        
        finite_predictions = predictions[isfinite(predictions)]
        if len(finite_predictions) == 0:
            return 0.0, gini_from_value(0.0)
        
        result = minimize_scalar(gini_from_value, bounds=(finite_predictions[0],finite_predictions[-1]))
        if result.success:
            return result.x, gini_from_value(result.x)

    def fit(self, generations=100, population_size=200):
        random.seed(41)
        pop = self.toolbox.population(n=population_size)
        if self.algorithm == 'eaSimple':
            algorithms.eaSimple(pop, self.toolbox, cxpb=0.5, mutpb=0.2, ngen=generations, verbose=False)
        else:
            algorithms.eaMuPlusLambda(pop, self.toolbox, mu=population_size, lambda_=population_size, cxpb=0.5, mutpb=0.3, ngen=generations, verbose=False)

        # Return the best individual
        self.best = tools.selBest(pop, 1)[0]
        self.best_function = self.toolbox.compile(expr=self.best)
        if not self.arithmetic:
            self.threshold = None
        else:
            self.threshold, _ = self.findBestThreshold(self.best_function)