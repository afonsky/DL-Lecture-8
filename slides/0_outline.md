---
zoom: 0.82
---

# From Earlier Lectures to Today

<div class="grid grid-cols-2 gap-12">
<div>

### We already know
* Forward → loss → backward → update *(Lecture 2)*
* Learning-rate search and weight decay *(Lecture 2)*
* Backpropagation and initialization *(Lecture 3)*
* Batch normalization and residual connections *(Lecture 5)*
* Gradient clipping *(Lecture 6)*

</div>
<div>

### Plan for today
1. **The objective**<br><small>training loss, new-data performance, landscapes</small>
2. **Stochastic gradients**<br><small>noise, batch size, computation</small>
3. **Better update rules**<br><small>momentum, adaptive rates, Adam(W)</small>
4. **Training practice**<br><small>schedules, choice, debugging</small>
5. **Quiz**<br><small>five questions, about 10–15 minutes</small>

</div>
</div>

<br>

#### The organizing question: **where does the next parameter update come from?**

<span class="refs">Main text: [d2l.ai, Chapter 12](https://d2l.ai/chapter_optimization/index.html) · Practical companion: [A. Karpathy, CS231n: Training Neural Networks III](https://cs231n.github.io/neural-networks-3/) · [S. Raschka, Common Optimization Algorithms](https://sebastianraschka.com/pdf/lecture-notes/stat479ss19/L12_optim_slides.pdf)</span>

<!--
Timing plan (80 min, of which 10–15 min is the quiz):
- Opening and outline:                                      3 min
- 1. Objective and gradient descent (5 slides):           10 min
- 2. Stochastic gradients (5 slides):                     11 min
- 3. Update rules (9 slides, incl. playground):           22 min
- 4. Schedules and practice (6 slides):                   11 min
- Conclusions:                                               3 min
                                                 total   60 min + quiz

If running late, cut in this order:
(a) the convexity slide, (b) the Adam bias-correction slide (one sentence instead), (c) the AdamW slide (the code
only), (d) the scheduler code slide, (e) shorten the playground to SGD versus Momentum at eta=0.6.
-->
