# 🌀 deepfractal

Deep learning math iterated across the complex plane. The fractal geometry that emerges reflects how networks actually behave; where they converge, where they explode, and what sits at the edge between the two.

---

## Fractals

### 1. Forward Pass Dynamics
**`forward_pass.py`**

Iterates a single neural layer `z → tanh(w·z + b)` as a complex map, where `w` is the weight, `z` is the input, and `b` is the bias. `tanh` is the activation function, limiting the output range between -1 and 1. Each pixel represents a (weight, bias) pair. Color encodes whether the repeated forward pass diverges or stabilizes, and how fast.
