# Logistic Regression from Scratch

Implementation of Logistic Regression with L1 and L2 regularization using only NumPy — no sklearn, no shortcuts.

---

## What I Built

- **Binary classification** using the sigmoid activation function
- **Gradient descent** training loop from scratch — forward pass, cost calculation, gradient computation, weight update
- **L1 regularization (Lasso)** — penalizes large weights, pushes some weights to zero
- **L2 regularization (Ridge)** — penalizes large weights, keeps all weights small
- Tested on the **Breast Cancer Wisconsin dataset** (569 samples, 30 features)

---

## Bug I Debugged — Shape Mismatch

During training, I kept getting a shape mismatch error between arrays of shape `(569,1)` and `(569,)`.

**Root cause:** NumPy treats `(569,1)` (2D column vector) and `(569,)` (1D array) differently, causing matrix operations to break.

**Fix:** Used `.squeeze()` or `.reshape(-1)` to consistently keep arrays as 1D:

```python
y = y.reshape(-1)   # ensure shape is (569,) not (569,1)
```

**What I learned:** Always check and align array shapes before any matrix operation — shape bugs are silent and cause wrong results without always throwing errors.

---

## Cost vs Iterations

![Cost Curve](cost_curve.png)

Cost decreases smoothly across iterations confirming the model is converging correctly.

---

## Validation Against Sklearn

| | My Implementation | Sklearn |
|---|---|---|
| Accuracy | XX% | XX% |
| Precision | XX% | XX% |
| Recall | XX% | XX% |

Results match closely, confirming the implementation is correct.

---

## Files

| File | Description |
|---|---|
| `logistic_regression.py` | Core implementation |
| `notebook.ipynb` | Full walkthrough with plots and sklearn comparison |

---

## Libraries Used

- NumPy
- Matplotlib
- Scikit-learn (for validation only)
