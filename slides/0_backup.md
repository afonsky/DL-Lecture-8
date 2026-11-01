---
layout: section
---

# Backup Slides

Optional derivations and reference tables.

<!--
Not part of the 60-minute lecture plan. Use after questions or for self-study.
-->

---
zoom: 0.78
---

# Backup: Adam, Fully Written

<div class="grid grid-cols-[4fr_3fr] gap-8">
<div>

### 1. First and second moments
$$m_t=\beta_1m_{t-1}+(1-\beta_1)g_t$$
$$s_t=\beta_2s_{t-1}+(1-\beta_2)g_t^2$$

### 2. Correct zero initialization
$$\hat m_t=\frac{m_t}{1-\beta_1^t},\qquad \hat s_t=\frac{s_t}{1-\beta_2^t}$$

### 3. Parameter update
$$\theta_{t+1}=\theta_t-\eta\frac{\hat m_t}{\sqrt{\hat s_t}+\epsilon}$$

</div>
<div>

### Extra optimizer state
| Optimizer | Per parameter tensor |
|---|---|
| SGD | none |
| Momentum | velocity: **1×** |
| Adagrad | accumulated scale: **1×** |
| RMSProp | moving scale: **1×** |
| Adam / AdamW | first + second moment: **2×** |

<br>

For a model with 100 M parameters in float32, Adam's two states alone are about **800 MB**, before parameters, gradients and activations.

</div>
</div>

<span class="refs">Source: [d2l.ai 12.10.1](https://d2l.ai/chapter_optimization/adam.html#the-algorithm) · [Kingma & Ba, 2015, Algorithm 1](https://arxiv.org/abs/1412.6980)</span>

<!--
The state memory is 2 tensors x 100,000,000 values x 4 bytes = 800,000,000 bytes (about 763 MiB). Mixed-precision
training can change the dtype of parameters and states; framework details matter, so treat this as a float32 example.

Notation follows the slides: s for second moment, not v, to avoid confusion with momentum velocity.
-->

---
zoom: 0.84
---

# Backup: Nesterov Momentum and NAdam

<div class="grid grid-cols-2 gap-10">
<div>

### Nesterov momentum
Standard momentum evaluates the gradient at the current parameters, then follows accumulated velocity.

**Nesterov's variant:** look ahead along the velocity first, then evaluate the gradient there.

```python
optimizer = torch.optim.SGD(
    model.parameters(), lr=0.01,
    momentum=0.9, nesterov=True,
)
```

* Often improves the response to changing curvature
* Not a universal improvement; tune as a variant of momentum

</div>
<div>

### NAdam
**NAdam = Adam with a Nesterov-style momentum update**, keeping adaptive second moments.

```python
optimizer = torch.optim.NAdam(
    model.parameters(), lr=1e-3,
)
```

* Same broad use cases as Adam
* Useful as a comparison in an established recipe, not as an automatic upgrade
* For RNNs, gradient clipping may still matter *(Lecture 6)*

</div>
</div>

<span class="refs">Read: [A. Karpathy, CS231n: Nesterov momentum](https://cs231n.github.io/neural-networks-3/#sgd) · [Sutskever et al., On the Importance of Initialization and Momentum](https://proceedings.mlr.press/v28/sutskever13.html) · [Dozat, Incorporating Nesterov Momentum into Adam](https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ)</span>

<!--
This is deliberately an intuition and API slide, not another dense update equation. Nesterov's look-ahead is the only
new idea. NAdam combines it with Adam's second-moment scaling.

The original open review page is the source for NAdam. Do not claim that NAdam is particularly best for RNNs as a
general rule; that was too strong in the old deck.
-->

---
zoom: 0.75
---

# Backup: Architecture-Specific Starting Points

| Architecture / setting | Common starting recipe | Typical LR range to try | Important companions |
|---|---|---|---|
| Small MLP | Adam | $10^{-3}$, $3\times10^{-4}$ | Overfit one batch; standardize inputs |
| CNN from scratch | SGD + momentum 0.9 | $0.01$–$0.1$ | Step/cosine schedule; augmentation |
| RNN / LSTM | Adam | $10^{-3}$ | Gradient clipping; smaller LR if unstable |
| Transformer from scratch | AdamW | $10^{-4}$–$10^{-3}$ | Warmup + decay; decoupled weight decay |
| Fine-tuning a pretrained model | AdamW | $10^{-5}$–$10^{-4}$ | Much smaller LR; parameter groups |
| Sparse embeddings | Adagrad or Adam | recipe-dependent | Per-coordinate updates |

<v-click>

### These ranges are **search intervals, not correct answers**. Model scale, batch size, data and schedule can shift the best rate by orders of magnitude.

</v-click>

<span class="refs">Cross-check with [CS231n: parameter updates](https://cs231n.github.io/neural-networks-3/#sgd), [Y. Bengio, Practical Recommendations](https://arxiv.org/abs/1206.5533), and the original model's training recipe</span>

<!--
This replaces the old "optimizer x architecture cheat sheet", which presented single values and architecture rules
as settled facts. For reproduction, start with the original paper or repository rather than this table.

GANs and other adversarial objectives are outside this lecture; their optimization has additional two-player
stability issues and should not be reduced to the same one-line rules.
-->
