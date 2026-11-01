---
layout: section
---

# 1. The Objective and Gradient Descent

Training loss is not the final goal — and the gradient is only local information.

<!--
One minute. The section separates two questions that are often conflated: can the optimizer reduce training loss,
and does the resulting model work on new data? Then it recalls only the gradient-descent facts needed later.
-->

---
zoom: 0.86
---

# Optimization Is Not Generalization

<div class="grid grid-cols-[3fr_4fr] gap-8">
<div>

### The optimizer sees
* A **training loss**, averaged over finite data
* The **gradient at the current parameters**
* Nothing about future examples

<v-click>

### We actually want
* Low loss on **new data**
* So tune on **validation**, report on **test** *(Lecture 2)*
* A lower training loss does not guarantee a better model

</v-click>
</div>
<div>
  <figure>
    <img src="/optimization_goal.png" style="width: 480px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: relative; top: 10px">Image source: <a href="https://d2l.ai/chapter_optimization/optimization-intro.html#goal-of-optimization">d2l.ai Fig. 12.1.1 The minimum of the empirical risk may differ from the minimum of the risk</a></figcaption>
  </figure>
</div>
</div>

<span class="refs">Read: [d2l.ai 12.1.1](https://d2l.ai/chapter_optimization/optimization-intro.html#goal-of-optimization) · [Y. Abu-Mostafa, Learning From Data, Lecture 13: Validation](https://www.youtube.com/watch?v=o7zzaKd0Lkk)</span>

<!--
The curves are d2l's toy example, recomputed for this deck. The orange minimizer is best on training data; the blue
one is best on new data. This is not a fitted neural network; it illustrates two different objectives.

Connect to Lecture 2's train/validation/test roles and Lecture 3's early stopping. An optimizer minimizes the first;
model selection needs the second. Do not call the validation curve "population risk" in a real experiment.
-->

---
zoom: 0.87
---

# Where First-Order Optimization Can Stall

<img src="/landscape_challenges.png" style="width: 850px !important;">

<figcaption style="color:#b3b3b3ff; font-size: 11px; position: relative; top: 10px">Image source: <a href="https://d2l.ai/chapter_optimization/optimization-intro.html#optimization-challenges-in-deep-learning">d2l.ai Figs. 12.1.2, 12.1.3 and 12.1.5 Local minimum, saddle point and vanishing gradient</a></figcaption>

<div class="grid grid-cols-3 gap-5 mt-4">
<div>

**Local minimum**<br>Gradient is zero, but another valley may be lower.
</div>
<div>

**Saddle point**<br>Gradient is zero; some directions curve up, others down.
</div>
<div>

**Plateau**<br>Gradient is merely tiny, so progress can be extremely slow.
</div>
</div>

<v-click>

#### In high dimensions, saddle points and plateaus are central concerns; finding *a useful solution* matters more than certifying the global minimum.

</v-click>

<span class="refs">Read: [d2l.ai 12.1.2](https://d2l.ai/chapter_optimization/optimization-intro.html#optimization-challenges-in-deep-learning) · [Y. Bengio et al., Deep Learning, Ch. 8](https://www.deeplearningbook.org/contents/optimization.html)</span>

<!--
All three panels use d2l's functions: x cos(pi x), x^3 and tanh(x). Strictly, x^3 at zero is an inflection point;
a higher-dimensional saddle has both upward and downward directions.

Vanishing gradients through depth were measured in Lecture 3, and through time in Lecture 6. Initialization,
normalization and residual connections address parts of that problem; an optimizer alone does not cure it.
-->

---
zoom: 0.76
---

# Convex and Non-Convex: What Changes?

<div class="grid grid-cols-2 gap-10">
<div>

### Convex objective
* Every local minimum is a **global minimum**
* With suitable step sizes, gradient methods have convergence guarantees
* Linear, logistic and softmax regression *(Lecture 2)*

<v-click>

**Useful as a test bed** — not as a description of deep networks.
</v-click>
</div>
<div>

### Non-convex objective
* Many stationary points and differently shaped valleys
* No general global-optimality guarantee
* MLPs, CNNs, RNNs and Transformers

<v-click>

**Useful solutions still exist.** Architecture and initialization shape the optimization problem *(Lectures 3 and 5)*.
</v-click>
</div>
</div>

<div class="grid grid-cols-[2fr_3fr] gap-8 mt-3">
<div>

#### A guarantee is about convergence,<br>**not** about test accuracy.
</div>
<div>
  <figure>
    <img src="/VGG_Resnet_loss_landscapes.png" style="width: 270px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px;">2-D visualizations of VGG and ResNet loss surfaces. Source: <a href="https://arxiv.org/abs/1712.09913">Li et al., 2018, Fig. 1</a></figcaption>
  </figure>
</div>
</div>

<!--
Do not say "gradient descent always finds the best solution" even for convex objectives: step size and stopping
criteria matter, and a convex function need not attain a minimum. The key property is local minimum = global minimum.

The figure is a 2-D visualization of a very high-dimensional surface, not the surface itself. Residual connections
make this visualization smoother; Lecture 5 showed their effect on trainability. Skip this slide first if short of time.
-->

---
zoom: 0.85
---

# Gradient Descent: One Update, Two Decisions

<div class="grid grid-cols-[4fr_3fr] gap-8">
<div>

### Direction and step size
$$\theta_{t+1} = \theta_t - \eta\,g_t$$

* $g_t=\nabla_\theta\mathcal L(\theta_t)$: local direction of steepest **ascent**
* Move in the **opposite** direction
* $\eta$: the **learning rate**, chosen by us

### PyTorch: the optimizer owns the update
```python
optimizer.zero_grad()
logits = model(X)
loss = loss_fn(logits, y)
loss.backward()       # autograd computes every g_t
optimizer.step()      # the optimizer applies its update rule
```

</div>
<div>
  <figure>
    <img src="/gradient_descent_2.png" style="width: 360px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: relative; top: 10px">Image source: <a href="https://d2l.ai/chapter_optimization/gd.html#multivariate-gradient-descent">d2l.ai Fig. 12.3.5 Gradient descent on a two-dimensional quadratic objective</a></figcaption>
  </figure>
</div>
</div>

<span class="refs">Review: training loop in **Lecture 2**, backpropagation in **Lecture 3** · [J. Starmer, StatQuest: Gradient Descent](https://www.youtube.com/watch?v=sDv4f4s2SB8) · [d2l.ai 12.3](https://d2l.ai/chapter_optimization/gd.html)</span>

<!--
Backpropagation and the optimizer are separate: backward() computes gradients; step() changes parameters. The rest of
today changes only the constructor of optimizer and, sometimes, the learning rate during training.

The loop omits device transfers and model.train() for focus; it is the update part of Lecture 2's complete loop.
-->

---
zoom: 0.86
---

# Learning Rate: The First Thing to Tune

<img src="/learning_rate_gd.png" style="width: 850px !important;">

<figcaption style="color:#b3b3b3ff; font-size: 11px; position: relative; top: 10px">Image source: <a href="https://d2l.ai/chapter_optimization/gd.html#learning-rate">d2l.ai Figs. 12.3.1–12.3.3 Gradient descent on f(x) = x² with learning rates 0.2, 0.05 and 1.1</a></figcaption>

<div class="grid grid-cols-[3fr_2fr] gap-8 mt-3">
<div>

* **Too small:** many updates, slow progress
* **Too large:** overshoot, oscillation or divergence
* **Just right:** fast early progress and stable later progress
</div>
<div>

#### Practical search
Try rates on a **log scale**<br>(…, 0.01, 0.001, 3e-4, …), on validation data *(Lecture 2)*.
</div>
</div>

<span class="refs">Read: [Y. Bengio, Practical Recommendations for Gradient-Based Training](https://arxiv.org/abs/1206.5533) · [A. Ng, Improving Deep Neural Networks, Week 2](https://www.coursera.org/learn/deep-neural-network)</span>

<!--
The right panel uses a log y-axis so that divergence remains visible. After ten steps, x is about 61.9 for eta=1.1,
3.49 for eta=0.05 and 0.060 for eta=0.2 — the same calculations as d2l.

Typical starting points are starting points, not laws: SGD 0.1 or 0.01; Adam 0.001 or 3e-4. Search multiplicatively,
not by adding 0.001. Lecture 2 already introduced the learning-rate range test.
-->
