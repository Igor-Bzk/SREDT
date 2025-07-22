from sklearn.metrics import accuracy_score, confusion_matrix
from SREDT.SREDTClassifier import SREDTClassifier
import sys


def load_circles():
    from sklearn.datasets import make_circles
    return make_circles(n_samples=100, noise=0.1, factor=0.5, random_state=42)

def load_simple_data():
    from numpy import array, linspace, column_stack
    X1 = linspace(-5, 5, 100)
    X2 = linspace(-5, 5, 100)
    X3 = array([-1,1]*50)
    X_features = column_stack((X1, X2, X3))
    y = array([2 if x3 < 0 else 0 if x1*x2 < 2 else 1 for x1, x2, x3 in X_features])
    return X_features, y

def test_SREDT(dataset='circles'):

    if dataset == 'circles':
        X_features, y = load_circles()
    elif dataset == 'simple':
        X_features, y = load_simple_data()
    else:
        raise ValueError("Unknown dataset")

    test = SREDTClassifier(generations=100, population_size=20, function_set=('add', 'mul', 'square'))
    test.fit(X_features, y)
    predictions = test.predict(X_features)    
    print("Accuracy:", accuracy_score(y, predictions))
    print("Confusion Matrix:")
    print(confusion_matrix(y, predictions))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
    else:
        dataset_name = 'circles'
    test_SREDT(dataset=dataset_name)
