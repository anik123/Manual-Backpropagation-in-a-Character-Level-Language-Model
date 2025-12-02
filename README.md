# 🧠 Manual Backpropagation in a Character-Level Language Model

*A from-scratch implementation of forward and backward passes in PyTorch — without using autograd.*

This project builds a complete character-level language model (MLP) and implements **every gradient calculation manually**, including embeddings, linear layers, activations, softmax cross-entropy, and batch normalization.
It closely follows the deep-dive principles from Andrej Karpathy’s *Neural Networks: Zero to Hero* series.

---

## 🚀 Overview

Modern deep learning frameworks abstract away backpropagation, but understanding how gradients actually flow is essential for building strong intuition.

This repository demonstrates:

* How to construct a dataset of name strings for next-character prediction
* How to build an embedding + MLP architecture from scratch
* How to perform every step of the **forward pass manually**
* How to compute the **backward pass manually** (no `loss.backward()`)
* How to **validate gradients** against PyTorch's autograd
* How to train the model using **only custom gradients**
* How to generate new names using the trained model

---

## 🎯 Key Learning Objectives

### ✔ Manual Backpropagation

You will compute gradients for:

* Embedding lookup
* Fully connected layers
* Tanh activation
* Softmax + cross-entropy
* **Batch Normalization** (including the full backward expression)

### ✔ Gradient Verification

The project includes utilities to compare your manual gradients with PyTorch’s autograd:

```python
cmp('W1', dW1, W1)
cmp('counts', dcounts, counts)
cmp('bnvar_inv', dbnvar_inv, bnvar_inv)
```

These checks ensure bit-level correctness.

### ✔ Training With Custom Gradients

Once manual gradients match autograd’s, the entire training loop updates parameters using:

```python
p.data -= lr * grad
```

No `p.grad`, no `.backward()`.


---

## 🧪 Example Output (Generated Names)

```
maro.
lenia.
joren.
kalani.
tamor.
```

Every run generates unique samples depending on the seed.

---

## 📚 Concepts Covered

* Character embeddings
* Context windows for sequence modeling
* Matrix math behind forward propagation
* Manual calculation of Jacobians and gradients
* Batch normalization (forward & backward)
* Softmax stabilization tricks
* Training loops & learning rate scheduling
* Autoregressive text sampling

---

