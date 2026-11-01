---
layout: section
---

# Better Update Rules

Add memory of past gradients — for direction, for scale, or for both.

<!--
Two orthogonal ideas organize the section. Momentum averages the gradient direction. Adaptive methods track squared
gradients to rescale each coordinate. Adam keeps both kinds of state; AdamW changes how weight decay is applied.
-->

---
zoom: 0.86
---

# The Problem: One Learning Rate, Very Different Directions

<img src="/momentum_ravine.png" style="width: 950px !important;">

<figcaption style="color:#b3b3b3ff; font-size: 11px; position: relative; top: 10px">Image source: <a href="https://d2l.ai/chapter_optimization/momentum.html#an-ill-conditioned-problem">d2l.ai Figs. 12.6.1–12.6.3 Gradient descent and momentum on an ill-conditioned quadratic objective</a></figcaption>

<div class="grid grid-cols-[4fr_2fr] gap-8 mt-3">
<div>

* $f(x_1,x_2)=0.1x_1^2+2x_2^2$: flat along $x_1$, steep along $x_2$
* A safe rate makes progress slowly; a larger rate oscillates or diverges
* **Momentum** averages consistent directions and cancels alternating ones
</div>
<div>

> #### Key word: **preconditioning**<br>adapt the effective step to the geometry of the problem.
</div>
</div>

<span class="refs">Read: [d2l.ai 12.6](https://d2l.ai/chapter_optimization/momentum.html) · Interactive: [G. Goh, Why Momentum Really Works, Distill](https://distill.pub/2017/momentum/)</span>

<!--
The trajectories reproduce d2l's twenty updates. With eta=0.6 plain GD leaves the plotted region; momentum with
beta=0.5 reaches the minimum at the same rate. This is a quadratic demonstration, not a promise for every network.

Ill-conditioned means very different curvature in different directions. Input standardization (Lecture 5) and
normalization layers make some instances easier; adaptive methods estimate coordinate-wise scaling from gradients.
-->

---
zoom: 0.99
---

# Momentum: Average the Direction Over Time

<div class="grid grid-cols-[4fr_3fr] gap-9">
<div>

### Update
$$v_t=\beta v_{t-1}+g_t,\qquad \theta_{t+1}=\theta_t-\eta v_t$$

* $v_t$: **velocity**, a leaky average of past gradients
* $\beta=0$ recovers plain SGD;<br> $\beta=0.9$ is a common default
* Consistent gradients accumulate;<br> alternating gradients cancel

</div>
<div>

### What changes in PyTorch?
```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
)
```

* One extra tensor per parameter tensor
* Retune $\eta$: the accumulated step scale is about $1/(1-\beta)$

</div>
</div>

<div class="grid grid-cols-2 gap-8 mt-4">
<div>

> #### Like a ball downhill
> Builds speed when the slope keeps pointing the same way.
</div>
<div>

> #### Unlike a bigger batch
> Averages over **time**, not over more examples at one step.
</div>
</div>

<span class="refs">Read: [d2l.ai 12.6.1](https://d2l.ai/chapter_optimization/momentum.html#basics) · [A. Ng, Improving Deep Neural Networks, Week 2: Momentum](https://www.coursera.org/learn/deep-neural-network) · [A. Karpathy, CS231n: parameter updates](https://cs231n.github.io/neural-networks-3/#sgd)</span>

<!--
Convention warning: in this equation the new gradient has coefficient 1, not (1-beta). The history weights sum to
1/(1-beta), so the step magnitude changes; the old slide incorrectly called the current gradient's weight 10% at
beta=0.9. An EMA convention with coefficient (1-beta) is different and appears in Adam.

Momentum does not guarantee escape from every local minimum. Its demonstrated benefits here are damping oscillations
and accelerating consistent progress.
-->

---
zoom: 0.99
---

# Try It: Which Update Reaches the Minimum?

<OptimizerPlayground />

<br>

<span class="refs">Same objective and update rules as [d2l.ai 12.6](https://d2l.ai/chapter_optimization/momentum.html) and [12.8–12.10](https://d2l.ai/chapter_optimization/adam.html) · Demo runs locally in the slide</span>

<!--
Suggested 3-minute sequence:
1. Momentum, eta=0.6, beta=0: +20 steps. It is plain GD and diverges.
2. Reset, beta=0.5: +20 steps. Same rate, now it converges. Ask which direction was damped and which accelerated.
3. Try beta=0.9: the longer memory overshoots more; momentum is not "always better".
4. Switch to RMSProp or Adam at eta=0.1: the coordinate-wise scale changes the trajectory.

The demo is deterministic and needs no network. It stops after 50 steps or divergence. In the printed deck the
previous slide supplies the static comparison.
-->

---
zoom: 0.97
---

# Adagrad: A Different Rate for Every Coordinate

<div class="grid grid-cols-[4fr_3fr] gap-8">
<div>

### Motivation: sparse features
A common word's embedding is updated often; a rare word's embedding only occasionally. One global decay schedule fits both badly.

### Update
$$s_t=s_{t-1}+g_t^2,\qquad \theta_{t+1}=\theta_t-\frac{\eta}{\sqrt{s_t+\epsilon}}\,g_t$$

</div>
<div>

### Interpretation
* Operations are **coordinate-wise**
* Large accumulated gradients → smaller effective rate
* Small or infrequent gradients → relatively larger steps
* One extra state tensor per parameter tensor

</div>
</div>

<v-click>

### The weakness
* $s_t$ only grows, so the effective rate can collapse too early.
* Useful for sparse problems; often too aggressive for long deep-learning runs.

</v-click>

<span class="refs">Read: [d2l.ai 12.7](https://d2l.ai/chapter_optimization/adagrad.html) · Paper: [Duchi, Hazan & Singer, 2011](https://jmlr.org/papers/v12/duchi11a.html) · [A. Karpathy, CS231n: adaptive methods](https://cs231n.github.io/neural-networks-3/#ada)</span>

<!--
Adagrad is a diagonal preconditioner estimated from squared gradients, not a full second-order method. It avoids the
O(d^2) Hessian of Newton's method.

The rare-word example follows d2l: frequent features may already be learned while rare ones still need updates.
The next slide replaces the ever-growing sum by a moving average.
-->

---
zoom: 0.89
---

# RMSProp: Forget Old Gradient Magnitudes

<div class="grid grid-cols-[4fr_3fr] gap-8">
<div>

### Replace the sum by a moving average
$$s_t=\gamma s_{t-1}+(1-\gamma)g_t^2$$
$$\theta_{t+1}=\theta_t-\frac{\eta}{\sqrt{s_t+\epsilon}}\,g_t$$

* Recent gradient magnitudes matter more
* $\gamma=0.9$ is typical; old history is exponentially downweighted
* The global learning rate remains under our control

</div>
<div>

### Adagrad → RMSProp
| | Adagrad | RMSProp |
|---|---|---|
| State | sum of $g^2$ | moving average of $g^2$ |
| Memory | all history | recent history |
| Rate over time | can vanish | need not vanish |

<br>

```python
optimizer = torch.optim.RMSprop(
    model.parameters(), lr=0.01, alpha=0.9
)
```

</div>
</div>

<span class="refs">Read: [d2l.ai 12.8](https://d2l.ai/chapter_optimization/rmsprop.html) · Original source: [G. Hinton, Neural Networks for Machine Learning, Lecture 6, slide 29](https://www.cs.toronto.edu/~hinton/coursera/lecture6/lec6.pdf#page=29)</span>

<!--
RMSProp was introduced in Hinton's Coursera slides rather than a paper. In PyTorch alpha is the decay coefficient
called gamma here. Note the API uses sqrt(s + eps); Adam uses sqrt(s_hat) + eps.

It still adapts each coordinate but does not accumulate an unbounded sum. It does not use momentum in its standard
form, although PyTorch exposes an optional momentum argument.
-->

---
zoom: 0.86
---

# Adam: Momentum + Adaptive Scaling

<div class="grid grid-cols-[4fr_3fr] gap-16">
<div>

### Keep two moving averages
| State | Tracks | Typical coefficient |
|---|---|---:|
| $m_t$ | gradient $g_t$ — direction | $\beta_1=0.9$ |
| $s_t$ | squared gradient $g_t^2$ — scale | $\beta_2=0.999$ |

### Update (after bias correction)
$$\theta_{t+1}=\theta_t-\eta\,\frac{\hat m_t}{\sqrt{\hat s_t}+\epsilon}$$

* $\hat m_t,\hat s_t$ correct the zero initialization
* Two extra state tensors per parameter tensor
* Full four-step equations: backup slide

</div>
<div>

### Defaults and code
| Hyperparameter | Common default |
|---|---:|
| $\eta$ | 0.001 |
| $\beta_1$ | 0.9 |
| $\beta_2$ | 0.999 |
| $\epsilon$ | $10^{-8}$ |

```python
optimizer = torch.optim.Adam(
    model.parameters(), lr=1e-3,
    betas=(0.9, 0.999), eps=1e-8,
)
```

**A strong first baseline — not a guarantee of the best validation score.**

</div>
</div>

<span class="refs">Read: [d2l.ai 12.10](https://d2l.ai/chapter_optimization/adam.html) · Paper: [Kingma & Ba, Adam, 2015](https://arxiv.org/abs/1412.6980) · [S. Raschka, Common Optimization Algorithms](https://sebastianraschka.com/pdf/lecture-notes/stat479ss19/L12_optim_slides.pdf)</span>

<!--
Adam's second-moment average is much slower (beta2=0.999) than its first-moment average (beta1=0.9). Adaptive
scaling does not eliminate learning-rate tuning.

This closes a course-long gap: Adam and AdamW were used in Lectures 2–7 without an explanation. The update looks
like RMSProp but uses the smoothed gradient m instead of the raw gradient g.
-->

---
zoom: 0.98
---

# Adam's Warm-Up Problem: Bias Correction

#### Both moving averages start at zero, so early estimates are biased toward zero.

<br>

<div class="grid grid-cols-[3fr_4fr] gap-8">
<div>


### First step, constant $g=1$
* $m_1=(1-0.9)\cdot1=0.1$, but $\hat m_1=0.1/(1-0.9)=1$
* $s_1=(1-0.999)\cdot1=0.001$, but $\hat s_1=0.001/(1-0.999)=1$

</div>
<div>
  <figure>
    <img src="/adam_bias.png" style="width: 500px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px;">Computed for this lecture from the bias-correction argument in <a href="https://d2l.ai/chapter_optimization/adam.html#the-algorithm">d2l.ai 12.10.1</a></figcaption>
  </figure>
</div>
</div>

<br>
<v-clicks>

* Correction divides by $1-\beta^t$; its effect fades as $t$ grows.
* Without correction, Adam would systematically underestimate both moments at the start of training.

</v-clicks>


<span class="refs">Read: [Kingma & Ba, 2015, Section 3: initialization bias correction](https://arxiv.org/abs/1412.6980) · [A. Karpathy, CS231n: Adam](https://cs231n.github.io/neural-networks-3/#ada)</span>

<!--
At t=1, m_hat=m/(1-beta1) and s_hat=s/(1-beta2), recovering the constant gradient's true moments 1 and 1.
With varying gradients, correction still removes the artificial preference for zero, not the sampling noise.

Skip the detailed numbers if late; keep the sentence "zero-initialized moving averages need bias correction".
-->

---
zoom: 0.92
---

# AdamW: Decouple Weight Decay from the Gradient

<div class="grid grid-cols-[4fr_3fr] gap-8">
<div>

### Adam + L2 penalty
Adding $\lambda\theta$ to the gradient sends weight decay through Adam's adaptive scaling. The effective decay then differs by coordinate.

### AdamW
Apply the adaptive update and the shrinkage separately:
$$\theta_{t+1}=\theta_t-\eta\left(\frac{\hat m_t}{\sqrt{\hat s_t}+\epsilon}+\lambda\theta_t\right)$$

* $\lambda$ controls **decoupled weight decay**
* This is the usual modern choice when weight decay is wanted

</div>
<div>

### PyTorch
```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.01,
)
```

<v-click>

* Revisit weight decay in **Lecture 2**
* Common practice: exclude biases and normalization parameters from decay using parameter groups
* AdamW ≠ "Adam with a smaller learning rate"

</v-click>
</div>
</div>

<span class="refs">Paper: [Loshchilov & Hutter, Decoupled Weight Decay Regularization, 2019](https://arxiv.org/abs/1711.05101) · [PyTorch AdamW documentation](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)</span>

<!--
Weight decay is a generalization choice, not a convergence cure. The original AdamW paper shows why L2 regularization
and decoupled decay are not equivalent under adaptive scaling.

Do not label Adam's weight_decay argument "wrong": its default is coupled L2 regularization, which is simply a
different algorithm. Recent PyTorch also offers more controls; the important distinction is coupled versus decoupled.
-->

---
zoom: 0.89
---

# One Task, Six Optimizers — Not a Leaderboard

<img src="/optimizer_comparison.png" style="width: 890px !important;">

<div class="grid grid-cols-[4fr_2fr] gap-8 mt-2">
<div>

* Same 2–16–16–1 MLP, initial weights and batches; **3 seeds**
* Epochs to 95% validation accuracy: **SGD 8–32; Momentum 3–10; Adagrad 1–4; RMSProp 3–6; Adam(W) 3–9**
* Final validation accuracy: **all ≥ 99.6%** — the visible difference here is mainly speed

</div>
<div>

#### Read the curves honestly
A fixed budget changes the winner. Learning rates and schedules interact with the optimizer. One synthetic task cannot rank algorithms in general.

</div>
</div>

<figcaption style="color:#b3b3b3ff; font-size: 11px;">Computed for this lecture: two-moons data, 1,024 train / 512 validation examples, batch 64, 100 epochs. Code and per-seed results: <code>scripts/lecture8/</code></figcaption>

<!--
Per-seed epochs to 95%: SGD [8, 32, 10]; Momentum [5, 10, 3]; Adagrad [3, 4, 1]; RMSProp [4, 6, 3];
Adam [6, 9, 3]; AdamW [6, 9, 3]. Final accuracies are in scripts/lecture8/results/optimizer_summary.json.

Recipes fixed in advance: SGD lr=0.1; Momentum lr=0.05, beta=0.9; Adagrad lr=0.1; RMSProp lr=0.01, gamma=0.9;
Adam(W) lr=0.01, betas=(0.9,0.999); AdamW decay=0.01. Validation data were not used to select a winner.
-->
