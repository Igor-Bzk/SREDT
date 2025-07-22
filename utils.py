import numpy as np

def gini(y_subset):
    if len(y_subset) == 0:
        return 0.0
    classes, counts = np.unique(y_subset, return_counts=True)
    probs = counts / len(y_subset)
    return 1.0 - np.sum(probs ** 2)

def split_gini(y_left, y_right):
    return (len(y_left) * gini(y_left) + len(y_right) * gini(y_right)) / (len(y_left) + len(y_right))

def splitSetOnFunction(func, X, y, threshold):
    if threshold is None:
        predictions = np.array([func(*features) for features in X], dtype=bool)
    else:
        predictions = np.array([func(*features) > threshold for features in X], dtype=bool)
    left_mask = predictions == False
    right_mask = predictions == True
    return X[left_mask], X[right_mask], y[left_mask], y[right_mask]

def readable_deap_function(expr):
    """
    Converts a DEAP PrimitiveTree expression to a readable string.
    E.g., add(x, y) -> x + y
    """
    class Node:
        def __init__(self, name, args=None, value=None):
            self.name = name
            self.args = args if args is not None else []
            self.value = value

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
        elif node.name == 'not_':
            return f"(!{node_to_str(node.args[0])})"
        elif node.name == 'xor':
            return f"({node_to_str(node.args[0])} ^ {node_to_str(node.args[1])})"
        elif node.name == 'sqrt':
            return f"(√{node_to_str(node.args[0])})"
        elif node.name == 'div':
            return f"({node_to_str(node.args[0])} / {node_to_str(node.args[1])})"
        elif isinstance(node.name, str) and node.name.startswith('ARG'):
            return node.name
        elif node.name == 'True':
            return "True"
        else:
            return str(node.name)
    

    def parse_tree(tree, pos=0):
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

    expr_node, _ = parse_tree(expr)
    return node_to_str(expr_node)