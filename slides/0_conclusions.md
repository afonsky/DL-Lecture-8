---
zoom: 0.78
---

# Conclusions: The Optimization Toolkit

<div class="grid grid-cols-2 gap-10">
<div>

### Ideas you will reuse
* **Mini-batch SGD:** unbiased gradients with noise $\propto1/\sqrt b$, plus efficient vectorization
* **Momentum:** average direction over time; damp oscillations, accelerate consistency
* **Adagrad / RMSProp:** scale each coordinate by its gradient history
* **Adam:** momentum + adaptive scaling + bias correction
* **AdamW:** decouple weight decay from the adaptive update
* **Schedules:** large enough steps early, small enough steps late

</div>
<div>

### Habits that matter more than names
* Tune the **learning rate first**, on a log scale
* Compare by **wall time, steps and validation**, not epochs alone
* Match initialization, data and budget when comparing optimizers
* Overfit one batch, monitor gradients, read train/validation curves
* Use several seeds before believing a small difference

</div>
</div>

<br>

#### The one idea to take away: **an optimizer is a rule for turning a noisy local gradient into a useful step — direction, scale and memory are separate design choices.**

<!--
One minute. There is no universal winner. The controlled example reached almost the same final validation accuracy
with all six methods but at very different speeds; tuning and the training budget decide what is practical.

Callbacks: the training loop (Lecture 2), weight decay (Lecture 2), initialization and overfit-one-batch (Lecture 3),
batch norm and the visible step schedule (Lecture 5), gradient clipping (Lecture 6).
-->

---
zoom: 0.76
---

# Learn More from the Experts

<div class="experts">

| Expert | Watch / read | Today's topics |
|---|---|---|
| Andrew Ng | [Improving Deep Neural Networks, Week 2: Optimization Algorithms](https://www.coursera.org/learn/deep-neural-network) | mini-batch SGD, momentum, RMSProp, Adam |
| Yaser Abu-Mostafa | [Learning From Data, Lecture 13: Validation](https://www.youtube.com/watch?v=o7zzaKd0Lkk) · [Lecture 17: Three Learning Principles](https://www.youtube.com/watch?v=EZBUDG12Nr0) | optimization versus generalization; honest model selection |
| Yann LeCun | [Efficient BackProp](https://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf) | stochastic learning, scaling, curvature and practical convergence |
| Yoshua Bengio | [Practical Recommendations for Gradient-Based Training](https://arxiv.org/abs/1206.5533) · [Deep Learning, Ch. 8](https://www.deeplearningbook.org/contents/optimization.html) | learning rates, debugging and optimization for deep models |
| Andrej Karpathy | [CS231n: Training Neural Networks III](https://cs231n.github.io/neural-networks-3/) · [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) | update rules, schedules, monitoring and debugging |
| Sebastian Raschka | [STAT 479, Lecture 12: Common Optimization Algorithms](https://sebastianraschka.com/pdf/lecture-notes/stat479ss19/L12_optim_slides.pdf) | LR decay, momentum and adaptive optimizers |
| Josh Starmer | [StatQuest: Gradient Descent](https://www.youtube.com/watch?v=sDv4f4s2SB8) · [Stochastic Gradient Descent](https://www.youtube.com/watch?v=vMh0zPT0tLI) | the core algorithms, step by step |

</div>

<style>
.experts td, .experts th { padding-top: 0.3em; padding-bottom: 0.3em; }
</style>

<br>

#### Main text: [d2l.ai, Chapter 12 — Optimization Algorithms](https://d2l.ai/chapter_optimization/index.html) · Interactive: [Why Momentum Really Works](https://distill.pub/2017/momentum/)

<!--
For self-study: Starmer first for the mechanics, Ng for the unified optimizer sequence, Raschka for a compact second
explanation, and CS231n for training practice. Bengio's chapter and LeCun's paper are the deeper references.

Abu-Mostafa does not have a deep-learning optimizer lecture; his validation material answers the equally important
question of how to choose among the models that optimization produces.
-->
