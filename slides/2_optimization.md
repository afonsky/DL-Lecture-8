---
layout: section
---

# Stochastic Gradients

Trade exact but expensive gradients for noisy but frequent updates.

<!--
The key distinction is statistical efficiency against computational efficiency. A single example gives a cheap,
noisy gradient; a large batch gives a reliable but expensive one. Mini-batches and vectorization sit between them.
-->

---
zoom: 0.98
---

# Full-Batch GD, SGD and Mini-Batch SGD

### All three implement the same update — only the gradient estimate changes:
$$\theta_{t+1}=\theta_t-\frac{\eta}{|\mathcal B_t|}\sum_{i\in\mathcal B_t}\nabla\ell(\theta_t;x_i,y_i)$$

<div class="grid grid-cols-3 gap-5 mt-4">
<div>

#### Full batch: $|\mathcal B_t|=n$
* Exact training-set gradient
* **1 update per epoch**
* $O(n)$ examples per update
* Low noise, poor update frequency
</div>
<div>

#### SGD: $|\mathcal B_t|=1$
* One random example
* **$n$ updates per epoch**
* $O(1)$ example per update
* Noisy, poor hardware utilization
</div>
<div>

#### Mini-batch: $1<|\mathcal B_t|\ll n$
* Average over a random subset
* **$n/b$ updates per epoch**
* $O(b)$ examples per update
* Noise reduction + vectorization
</div>
</div>

<v-click>

#### Full batch does **not** require storing every example's gradient: examples can be processed in chunks and their gradients summed or averaged.

</v-click>

<span class="refs">Read: [d2l.ai 12.4](https://d2l.ai/chapter_optimization/sgd.html) and [12.5](https://d2l.ai/chapter_optimization/minibatch-sgd.html) · [Y. LeCun et al., Efficient BackProp](https://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)</span>

<!--
The computational cost counts example evaluations, not just arithmetic: full batch sees n examples for one update;
SGD sees one. Mini-batch usually gives the best wall-clock compromise because matrix operations use the hardware well.

Memory correction: a full-batch implementation may stream through the data and accumulate the gradient. It does not
need one stored gradient tensor per example.
-->

---
zoom: 0.99
---

# Why Does a Noisy Gradient Work?

<div class="grid grid-cols-[4fr_3fr] gap-8">
<div>

### On average, it points the right way
For a uniformly sampled example $i$:
$$\mathbb E_i[\nabla\ell_i(\theta)]=\nabla\mathcal L(\theta)$$

* The estimate is **unbiased**, not exact
* Individual steps may increase the loss
* A constant learning rate leaves noise-driven jitter near a minimum

<v-click>

#### Therefore: reduce the learning rate during training — but not so quickly that early progress stops.

</v-click>
</div>
<div>
  <figure>
    <img src="/sgd.png" style="width: 350px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: relative; top: 10px">Image source: <a href="https://d2l.ai/chapter_optimization/sgd.html#stochastic-gradient-updates">d2l.ai Fig. 12.4.1 A noisy stochastic-gradient trajectory with a constant learning rate</a></figcaption>
  </figure>
</div>
</div>

<span class="refs">Read: [d2l.ai 12.4.1–12.4.2](https://d2l.ai/chapter_optimization/sgd.html) · [J. Starmer, StatQuest: Stochastic Gradient Descent](https://www.youtube.com/watch?v=vMh0zPT0tLI)</span>

<!--
Unbiased means correct after averaging over the random example choice, holding theta fixed. It does not mean that
one batch's gradient is close to the full gradient, or that every update decreases the loss.

The figure adds zero-mean noise to d2l's exact 2-D gradient to mimic SGD. The noisy endpoint motivates schedules
later. It is not evidence that noise always helps generalization.
-->

---
zoom: 0.94
---

# Mini-Batches Reduce Noise — with Diminishing Returns

<div class="grid grid-cols-[3fr_4fr] gap-8">
<div>

For $b$ independent example gradients with per-example standard deviation $\sigma$:

$$\operatorname{Std}(\bar g)=\frac{\sigma}{\sqrt b}$$

<v-clicks>

* $b:1\rightarrow16$: noise $1\rightarrow1/4$
* $b:16\rightarrow256$: another $16\times$ examples → noise $1/4\rightarrow1/16$
* Beyond some point, extra computation buys little extra precision

</v-clicks>
</div>
<div>
  <figure>
    <img src="/minibatch_noise.png" style="width: 430px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px;">Computed for this lecture from the minibatch variance argument in <a href="https://d2l.ai/chapter_optimization/minibatch-sgd.html#minibatches">d2l.ai 12.5.2</a></figcaption>
  </figure>
</div>
</div>

#### Choose a batch large enough for efficient computation and small enough to fit memory — not by a universal "best" number.

<span class="refs">Read: [d2l.ai 12.5.1–12.5.2](https://d2l.ai/chapter_optimization/minibatch-sgd.html) · [A. Ng, Improving Deep Neural Networks, Week 2: Mini-batch Gradient Descent](https://www.coursera.org/learn/deep-neural-network)</span>

<!--
The 1/sqrt(b) rule assumes independent examples with the same variance. Real batches are shuffled without
replacement and examples can be correlated, so treat the curve as the idealized guide from d2l, not a measured
property of every dataset.

Ask: why not make the batch enormous immediately? Each noise halving costs 4x the examples, and there are fewer
updates per epoch. Hardware utilization is the other side of the trade-off.
-->

---
zoom: 0.86
---

# Batches, Steps and Epochs in PyTorch

<div class="grid grid-cols-[4fr_3fr] gap-10">
<div>

### Vocabulary *(Lecture 2)*
* **Batch:** examples used for one update
* **Iteration / step:** one call to `optimizer.step()`
* **Epoch:** one pass through the training dataset

#### Example: CIFAR-10
| Quantity | Value |
|---|---:|
| Training images | $50\,000$ |
| Batch size | $100$ |
| Steps per epoch | $500$ |
| Steps in 100 epochs | $50\,000$ |

</div>
<div>
<br>

```python
train_loader = DataLoader(
    train_set,
    batch_size=100,
    shuffle=True,       # new order each epoch
)

for epoch in range(100):
    for X, y in train_loader:
        train_step(X, y)
```

* Shuffle **training**, not validation or test
* Without replacement: each example appears once per epoch

</div>
</div>

<v-click>

#### If batch size doubles at a fixed epoch count: examples seen stay fixed, but optimizer updates per epoch halve.

</v-click>

<span class="refs">Read: [d2l.ai 12.4.4: sampling without replacement](https://d2l.ai/chapter_optimization/sgd.html#stochastic-gradients-and-finite-samples) · [PyTorch DataLoader documentation](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)</span>

<!--
This is the calculation students should do before comparing runs. "Same number of epochs" does not mean the same
number of updates when batch sizes differ. For optimizer comparisons, report both epochs and steps, plus wall time.

Shuffling avoids a fixed order and gives a fresh permutation each epoch. With replacement, d2l notes that only about
63% of examples appear at least once in one pass of n draws.
-->

---
zoom: 0.93
---

# Choosing a Batch Size in Practice

<div class="grid grid-cols-2 gap-10">
<div>

### Smaller batch
* More updates per epoch
* Noisier gradients and loss curves
* May underuse the GPU
* Uses less activation memory

</div>
<div>

### Larger batch
* Better parallel utilization
* Less gradient noise
* Fewer updates per epoch
* Needs more activation memory; may require LR retuning

</div>
</div>

<v-click>

### Practical rules
* Compare by **wall-clock time and validation performance**, not batch size alone
* If memory is tight, **gradient accumulation** simulates a larger batch over several small ones
* Batch statistics also change batch normalization *(Lecture 5)* — very small batches can behave differently
* Noise can affect generalization, but **“smaller always generalizes better” is false**

</v-click>

<span class="refs">Read: [d2l.ai 12.5: vectorization and minibatches](https://d2l.ai/chapter_optimization/minibatch-sgd.html) · [A. Karpathy, CS231n: monitoring loss and batch noise](https://cs231n.github.io/neural-networks-3/#baby)</span>

<!--
Powers of two are convenient implementation choices, not an optimization theorem. Start with what fits and trains
efficiently (often 32–256 in small exercises), then measure.

Linear or square-root learning-rate scaling when changing batch size is a heuristic that works in some regimes, not
a rule to apply blindly. Accumulation changes update frequency, not the mathematical mean gradient (apart from details
such as batch norm and the final incomplete batch).
-->
