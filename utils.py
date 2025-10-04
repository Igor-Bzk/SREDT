from numpy import array, sum, bincount
from re import search

def gini(y_subset, nb_classes=None, distribution=False):
    if len(y_subset) == 0:
        if distribution:
            return 0.0, array([])
        return 0.0
    counts = bincount(y_subset, minlength=nb_classes) if nb_classes is not None else bincount(y_subset)
    probs = counts / len(y_subset)
    if distribution:
        return 1.0 - sum(probs ** 2), counts
    return 1.0 - sum(probs ** 2)

def split_gini(y_left, y_right, nb_classes=None):
    return (len(y_left) * gini(y_left, nb_classes=nb_classes) + len(y_right) * gini(y_right, nb_classes=nb_classes)) / (len(y_left) + len(y_right))

def splitSetOnFunction(func, X, y, threshold):
    if threshold is None:
        # If no threshold is provided, it is a logical function that returns boolean values.
        predictions = array([func(*features) for features in X], dtype=bool)
    else:
        predictions = array([func(*features) > threshold for features in X], dtype=bool)
    left_mask = predictions == False
    right_mask = predictions == True
    return X[left_mask], X[right_mask], y[left_mask], y[right_mask]

def readable_deap_function(expr, features=None):
    if features is not None:
        num_re = r'ARG(\d+)'
    """
    Converts a DEAP PrimitiveTree expression to a readable string.
    E.g., add(x, y) -> x + y
    """
    class Node:
        def __init__(self, name, args=None, value=None):
            self.name = name
            self.args = args if args is not None else []
            self.value = value

    def parse_tree(tree, pos=0):
        """
        Parses a DEAP PrimitiveTree and returns a Node object.
        """
        node = tree[pos]
        if hasattr(node, 'arity') and node.arity > 0:
            args = []
            next_pos = pos + 1
            for _ in range(node.arity):
                arg_str, next_pos = parse_tree(tree, next_pos)
                args.append(arg_str)
            return Node(node.name, args, getattr(node, 'value', None)), next_pos
        else:
            # Terminal node
            return Node(getattr(node, 'name', str(node)), [], getattr(node, 'value', None)), pos + 1
    
    def node_to_str(node):
        if node.name == 'add':
            return f"({node_to_str(node.args[0])} + {node_to_str(node.args[1])})"
        elif node.name == 'mul':
            return f"({node_to_str(node.args[0])} * {node_to_str(node.args[1])})"
        elif node.name == 'square':
            return f"({node_to_str(node.args[0])}²)"
        elif node.name == 'sub':
            return f"({node_to_str(node.args[0])} - {node_to_str(node.args[1])})"
        elif node.name == 'rand_float':
            return f"{node.value}"
        elif node.name == 'and_':
            return f"({node_to_str(node.args[0])} && {node_to_str(node.args[1])})"
        elif node.name == 'or_':
            return f"({node_to_str(node.args[0])} || {node_to_str(node.args[1])})"
        elif node.name == 'logical_not':
            return f"(!{node_to_str(node.args[0])})"
        elif node.name == 'xor':
            return f"({node_to_str(node.args[0])} ^ {node_to_str(node.args[1])})"
        elif node.name == 'sqrt':
            return f"(√{node_to_str(node.args[0])})"
        elif node.name == 'div':
            return f"({node_to_str(node.args[0])} / {node_to_str(node.args[1])})"
        elif isinstance(node.name, str) and node.name.startswith('ARG'):
            return node.name if features is None else features[int(search(num_re, node.name).group(1))]
        else:
            return str(node.name)
        
    return node_to_str(parse_tree(expr)[0])

def pickle(clf, filename):
    from pickle import dump
    from SREDT.SREDTClassifier import SREDT_leaf
    
    params = clf.get_params()

    def node_to_dict(node):
        if isinstance(node, SREDT_leaf):
            return dict(
                majority_class=node.majority_class,
                gini=node.gini if hasattr(node, 'gini') else None,
                class_distribution=node.class_distribution if hasattr(node, 'class_distribution') else None,
            )

        return dict(
            threshold=node.threshold,
            left=node_to_dict(node.left),
            right=node_to_dict(node.right),
            uncompiled_function=str(node.uncompiled_function),
            gini=node.gini if hasattr(node, 'gini') else None,
            class_distribution=node.class_distribution if hasattr(node, 'class_distribution') else None
        )

    root = node_to_dict(clf.root_)
    extra_attributes = {
        'arithmetic_': clf.arithmetic_,
        'n_features_in_': clf.n_features_in_,
    }
    
    with open(filename, 'wb') as f:
        dump((root, params, extra_attributes), f)

def unpickle(filename):
    from SREDT.SREDTClassifier import SREDT_node, SREDT_leaf
    # The creator has to be created and have attributes set before loading the classifier if that's not already done.
    from pickle import load
    
    try:
        with open(filename, 'rb') as f:
            clf = load(f)
    except:
        from dill import load as dl_load
        from deap import creator,base, gp
        if not hasattr(creator, 'FitnessMin'):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, 'Individual'):
            creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
        with open(filename, 'rb') as f:
            return dl_load(f)
    
    from SREDT.SymbolicClassifier import make_toolbox
    from deap.gp import PrimitiveTree
    from SREDT import SREDTClassifier
    if not isinstance(clf,tuple):
        return clf
    
    root, params, extra_attributes = clf
    reconstructed_clf = SREDTClassifier(params)
    toolbox = make_toolbox(params['function_set'], params['max_expression_height'], extra_attributes['n_features_in_'], arithmetic=extra_attributes['arithmetic_'])

    def dict_to_node(d):
        try:
            return SREDT_leaf(majority_class=d['majority_class'], gini=d.get('gini'), class_distribution=d.get('class_distribution'))
        except KeyError:
            uncompiled_tree = PrimitiveTree.from_string(d['uncompiled_function'], pset=toolbox.pset)
            return SREDT_node(
                function=toolbox.compile(expr=uncompiled_tree),
                threshold=d.get('threshold'),
                left=dict_to_node(d['left']),
                right=dict_to_node(d['right']),
                uncompiled_function=uncompiled_tree,
                arithmetic=extra_attributes['arithmetic_'],
                gini=d.get('gini'),
                class_distribution=d.get('class_distribution')
            )

    reconstructed_clf.root_ = dict_to_node(root)
    return reconstructed_clf