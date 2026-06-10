# Neural Network from Scratch — MNIST Digit Classifier

A 2-layer neural network built using **only NumPy**, trained on the MNIST handwritten digit dataset. No PyTorch, no TensorFlow — every forward pass, backward pass, and weight update is implemented by hand.

---

## Architecture

```
Input (784)  →  Hidden Layer (10, ReLU)  →  Output Layer (10, Softmax)
```

| Layer  | Neurons | Activation |
|--------|---------|------------|
| Input  | 784     | —          |
| Hidden | 10      | ReLU       |
| Output | 10      | Softmax    |

- **Loss function**: Cross-entropy  
- **Optimizer**: Vanilla gradient descent  
- **Data split**: 59,000 train / 1,000 dev  
- **Normalization**: pixel values scaled to [0, 1]

---

## What's Implemented from Scratch

- `init_params()` — small random weight initialization
- `forward_prop()` — Z = WX + b, then ReLU / Softmax
- `one_hot()` — label encoding for cross-entropy loss
- `backward_prop()` — full chain rule derivation (dZ2 → dW2 → dZ1 → dW1)
- `update_params()` — gradient descent parameter update
- `gradient_descent()` — full training loop with accuracy logging

---

## How to Run

```bash
# Install dependencies
pip install numpy matplotlib scikit-learn

# Run training
python neural_network.py
```

Training output (every 100 iterations):
```
Iteration    0 | Accuracy: 0.0821
Iteration  100 | Accuracy: 0.6543
Iteration  200 | Accuracy: 0.7891
Iteration  300 | Accuracy: 0.8340
Iteration  400 | Accuracy: 0.8612

Dev Set Accuracy: 0.8570
```

---

## Key Concepts

### Forward Propagation
```
Z1 = W1 · X + b1        # linear transform
A1 = ReLU(Z1)           # hidden layer activation
Z2 = W2 · A1 + b2       # linear transform
A2 = softmax(Z2)        # output probabilities
```

### Backpropagation (Chain Rule)
```
dZ2 = A2 - Y_one_hot               # softmax + cross-entropy derivative
dW2 = (1/m) · dZ2 · A1ᵀ
dZ1 = W2ᵀ · dZ2 * ReLU'(Z1)       # ReLU derivative: 1 if Z > 0
dW1 = (1/m) · dZ1 · Xᵀ
```

---

## Project Structure

```
ml-from-scratch/
├── neural_network.py      ← this file
├── linear_regression.py
├── logistic_regression.py
└── README.md
```

---

## Why From Scratch?

Understanding backpropagation at the matrix level — not just calling `.backward()` — is what separates someone who uses ML from someone who understands it. Every gradient here is derived and computed manually.

---

## Author

**Nithin** — CSE AI student building ML foundations from the ground up.  
Part of the [`ml-from-scratch`](.) series covering regression → classification → neural networks → transformers.
