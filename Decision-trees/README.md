# Decision Tree from Scratch — NumPy Only

A Decision Tree classifier built using **only NumPy**, implementing recursive binary splitting, information gain, and entropy — no scikit-learn under the hood.

---

## How It Works

A decision tree splits the dataset at each node by finding the **feature + threshold** combination that maximizes **information gain**. It keeps splitting recursively until it hits a stopping condition (max depth or min samples), then assigns the **majority class** as the leaf value.

```
Dataset
   │
   ├── feature_3 <= 1.75?
   │       ├── YES → feature_1 <= 5.0?
   │       │             ├── YES → class 0 🍃
   │       │             └── NO  → class 1 🍃
   │       └── NO  → class 2 🍃
```

---

## Architecture

### `Node`
Stores a single tree node — either a decision node (feature + threshold) or a leaf node (class value).

### `DecisionTree`
| Method | What it does |
|---|---|
| `fit(X, y)` | Builds the tree recursively from training data |
| `predict(X)` | Traverses the tree for each sample |
| `build_tree()` | Recursive splitting logic |
| `best_split()` | Tries every feature × threshold, picks best info gain |
| `split()` | Splits dataset into left (≤ threshold) and right (> threshold) |
| `information_gain()` | Parent entropy − weighted child entropy |
| `entropy()` | −Σ p·log₂(p) over class labels |

---

## The Math

### Entropy
Measures impurity in a node — 0 means perfectly pure, 1 means maximally mixed.

```
H(y) = -Σ p(c) · log₂(p(c))
```

### Information Gain
How much a split reduces impurity.

```
IG = H(parent) − [ (|left| / |parent|) · H(left) + (|right| / |parent|) · H(right) ]
```

The best split is the one with the **highest information gain**.

---

## How to Run

```bash
pip install numpy scikit-learn  # sklearn only used for dataset + train_test_split

python decision_tree.py
```

---

## Usage

```python
from decision_tree import DecisionTree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = DecisionTree(max_depth=3, min_samples_split=2)
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
accuracy = sum(predictions == y_test) / len(y_test)
print(f"Accuracy: {accuracy:.4f}")
```

---

## Hyperparameters

| Parameter | Default | Effect |
|---|---|---|
| `max_depth` | 2 | Controls tree depth — higher = more complex, risk of overfitting |
| `min_samples_split` | 2 | Minimum samples needed to split a node |

---

## Project Structure

```
ml-from-scratch/
├── neural_network.py
├── linear_regression.py
├── logistic_regression.py
├── decision_tree.py       ← this file
└── README.md
```

---

## Why From Scratch?

Calling `sklearn.tree.DecisionTreeClassifier` takes one line. Understanding *why* the tree splits where it does — entropy, information gain, recursive partitioning — requires building it yourself. Every split in this implementation is computed manually.

---

## Author

**Nithin** — CSE AI student building ML foundations from the ground up.  
Part of the [`ml-from-scratch`](.) series covering regression → classification → neural networks → transformers.
