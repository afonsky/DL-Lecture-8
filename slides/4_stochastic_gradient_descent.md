---
layout: section
---

# 4. Schedules and Training Practice

The optimizer chooses the update; the learning-rate schedule controls its size over time.

<!--
This section closes two earlier gaps: learning-rate schedules were named in Lecture 2 and used in Lecture 5 without
explanation. It finishes with an actionable choice and debugging workflow.
-->

---
zoom: 0.86
---

# Why Change the Learning Rate During Training?

<div class="grid grid-cols-2 gap-10">
<div>

### Early training
* Parameters are far from a useful solution
* Larger steps can make rapid progress
* Warmup may prevent unstable initial updates

</div>
<div>

### Late training
* Mini-batch noise keeps parameters moving
* Smaller steps allow finer settling
* Decay too quickly and learning stops prematurely

</div>
</div>

<v-click>

### Evidence already in this course
In **Lecture 5**, the learning rate was divided by 10 at epochs **82 and 123**. The sharp improvements in the ResNet curves at those points are the **schedule**, not a new architecture or optimizer.

</v-click>

<br>

#### A schedule is a plan for $\eta_t$; it is independent of whether the update uses SGD, momentum or Adam.

<span class="refs">Read: [d2l.ai 12.4.2: dynamic learning rate](https://d2l.ai/chapter_optimization/sgd.html#dynamic-learning-rate) and [12.11](https://d2l.ai/chapter_optimization/lr-scheduler.html) · [A. Karpathy, CS231n: annealing](https://cs231n.github.io/neural-networks-3/#anneal)</span>

<!--
Constant-rate SGD keeps a noise floor near a minimum (slide on stochastic gradients). Decay reduces that floor.
Warmup has the opposite purpose at the start: avoid destructive large steps while estimates and activations settle.

The Lecture 5 callback explains a feature students already saw in the CIFAR-10 curves. It does not claim every
learning-rate drop improves validation accuracy immediately or by a fixed amount.
-->

---
zoom: 0.80
---

# Four Useful Schedule Patterns

<div class="grid grid-cols-[4fr_3fr] gap-8">
<div>
  <figure>
    <img src="/lr_schedules.png" style="width: 510px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px;">Computed for this lecture after <a href="https://d2l.ai/chapter_optimization/lr-scheduler.html#policies">d2l.ai 12.11.3</a></figcaption>
  </figure>
</div>
<div>

| Schedule | When it is useful |
|---|---|
| **Constant** | Short baseline runs; diagnosing other problems |
| **Step decay** | Simple, interpretable; drop at chosen epochs or on plateau |
| **Cosine** | Smooth decay to a small final rate; common for long training |
| **Warmup + decay** | Unstable early training; standard for Transformers |

<v-click>

#### Choose and tune schedules on **validation data**. No schedule wins on every task.

</v-click>
</div>
</div>

<span class="refs">Read: [d2l.ai 12.11.3](https://d2l.ai/chapter_optimization/lr-scheduler.html#policies) · [Loshchilov & Hutter, SGDR: cosine restarts](https://arxiv.org/abs/1608.03983) · [S. Raschka, Common Optimization Algorithms](https://sebastianraschka.com/pdf/lecture-notes/stat479ss19/L12_optim_slides.pdf)</span>

<!--
The plot normalizes the initial maximum rate to 1. Step drops to 0.3 at 40% and 0.1 at 70%; cosine ends at 0.02;
warmup takes 10% of training. These are illustrative settings, not recommended defaults.

ReduceLROnPlateau is the data-driven version of step decay: it reacts when validation stops improving. Warmup is
particularly common for Transformers; it is not mandatory for every MLP or CNN.
-->

---
zoom: 0.85
---

# Schedulers in PyTorch: Step at the Right Time

<div class="grid grid-cols-[4fr_3fr] gap-8">
<div>

```python
optimizer = torch.optim.AdamW(
    model.parameters(), lr=1e-3, weight_decay=0.01
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=100
)

for epoch in range(100):
    train_one_epoch(model, train_loader, optimizer)
    scheduler.step()       # once per epoch, after optimizer updates
```

* `T_max` uses the same unit as the calls to `scheduler.step()`
* Log `optimizer.param_groups[0]["lr"]` to verify the schedule

</div>
<div>

### Alternatives
```python
scheduler = StepLR(
    optimizer, step_size=30, gamma=0.1
)
scheduler = ReduceLROnPlateau(
    optimizer, mode="min", factor=0.1, patience=5
)
```

<v-click>

#### `ReduceLROnPlateau` needs the validation metric: `scheduler.step(val_loss)`. A per-batch schedule needs `scheduler.step()` inside the batch loop instead.

</v-click>
</div>
</div>

<span class="refs">Read: [PyTorch lr_scheduler documentation](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate) · [d2l.ai 12.11.2: schedulers](https://d2l.ai/chapter_optimization/lr-scheduler.html#schedulers)</span>

<!--
The training helper must call optimizer.zero_grad(), backward() and optimizer.step(); scheduler.step() does not update
model parameters. Calling an epoch scheduler after every batch would make it run 500 times too fast on CIFAR-10.

For a warmup plus cosine schedule, use a library or compose schedulers; check the actual learning rates on a plot
before a long run. Skip this code slide if late; the previous slide carries the concepts.
-->

---
zoom: 0.70
---

# Which Optimizer Should You Start With?

| Situation | Reasonable starting point | Why / what to watch |
|---|---|---|
| General baseline, small/medium model | **Adam**, $\eta=10^{-3}$ | Fast, robust baseline; still tune $\eta$ |
| Weight decay, fine-tuning, Transformers | **AdamW**, $\eta=10^{-3}$ to $10^{-5}$ | Decoupled decay; warmup often important |
| CNN from scratch, longer tuned run | **SGD + momentum 0.9**, $\eta\approx0.1$ | Strong vision recipe; schedule and batch size matter |
| Sparse features / embeddings | **Adagrad** or **Adam** | Per-coordinate scaling helps infrequent updates |
| Existing RMSProp recipe | **RMSProp**, $\gamma=0.9$ | Keep it for a fair reproduction; not automatically better |
| Recurrent network | Adam(W) or SGD + gradient clipping | Clipping addresses spikes, not vanishing *(Lecture 6)* |

<v-click>

### These are **starting points, not laws**. Compare against a simple baseline with the same data, budget and evaluation.

</v-click>

<span class="refs">Guide: [A. Karpathy, CS231n: parameter updates](https://cs231n.github.io/neural-networks-3/#sgd) · [Y. Bengio, Practical Recommendations](https://arxiv.org/abs/1206.5533) · [A. Ng, Improving Deep Neural Networks, Week 2](https://www.coursera.org/learn/deep-neural-network)</span>

<!--
There is no universal "MLP uses X, CNN uses Y" rule. A well-tuned SGD recipe can beat Adam on vision; Adam(W) is a
convenient first choice in many other settings. Fine-tuning usually needs a smaller rate than training from scratch.

Do not compare a long SGD run with a short Adam run and conclude anything about final quality. Match the compute
budget and tune each method reasonably.
-->

---
zoom: 0.70
---

# Debugging: Read the Symptom Before Changing the Optimizer

| Symptom | First checks | Likely next action |
|---|---|---|
| Initial loss is wrong | Classes, labels, loss function | Expect about $\ln C$ for $C$-class uniform output *(Lecture 2)* |
| Cannot overfit one batch | Data/labels, shapes, gradients, model mode | Fix the pipeline before tuning *(Lecture 3)* |
| Loss becomes NaN / explodes | Learning rate, gradient norms, numerical stability | Lower LR; clip spikes; inspect data *(Lecture 6)* |
| Loss barely moves | LR too small, saturated units, frozen layers | Try larger LR; inspect gradients and initialization |
| Loss oscillates | LR, batch noise, bad batches | Lower LR or increase batch; check the data |
| Train ↓, validation ↑ | Overfitting, distribution shift | Regularization, augmentation, early stopping — not a new optimizer by default |
| Training is slow | Hardware utilization, batch size, data loading | Profile before changing the update rule |

<span class="refs">Guide: [A. Karpathy, A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) and [CS231n: babysitting training](https://cs231n.github.io/neural-networks-3/#baby) · [Y. Abu-Mostafa, Lecture 13: Validation](https://www.youtube.com/watch?v=o7zzaKd0Lkk)</span>

<!--
The table deliberately prevents "switch to Adam" from becoming the answer to every problem. An optimizer cannot fix
wrong labels, a target/loss mismatch, evaluation in train mode, or a broken data loader.

Gradient clipping limits the global gradient norm; it is useful for exploding gradients, not for a plateau caused by
vanishing gradients. Check for NaNs and bad inputs before assuming the learning rate is at fault.
-->

---
zoom: 0.86
---

# A Practical Optimization Workflow

<div class="grid grid-cols-2 gap-10">
<div>

### 1. Prove the pipeline
* Check initial loss and one batch of shapes
* **Overfit one small batch** *(Lecture 3)*
* Plot training and validation curves

### 2. Establish a baseline
* Simple optimizer, constant LR, fixed seed
* Try LR values on a **log scale**
* Record wall time and steps, not only epochs

</div>
<div>

### 3. Improve one thing at a time
* Add momentum or adaptive scaling
* Then tune schedule and weight decay on validation
* Repeat promising runs with several seeds

### 4. Keep evidence
* Log LR, gradient norms, losses and metrics
* Save the **best validation checkpoint**
* Change the test set only when you need a new unbiased evaluation *(Lecture 2)*

</div>
</div>

<v-click>

#### Debug first → baseline second → optimizer third → schedule fourth → repeat with seeds.

</v-click>

<span class="refs">Read: [A. Karpathy, A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) · [Y. Bengio, Practical Recommendations for Gradient-Based Training](https://arxiv.org/abs/1206.5533) · [Y. Abu-Mostafa, Lecture 13: Validation](https://www.youtube.com/watch?v=o7zzaKd0Lkk)</span>

<!--
This is the slide to photograph. A single seed can reverse a close optimizer comparison; our toy SGD reached 95%
validation in 8, 32 and 10 epochs. Report a range when the difference is small.

The final bullet means do not repeatedly tune on test data. If test data have influenced choices, they have joined
model selection and need replacement for an honest final estimate.
-->
