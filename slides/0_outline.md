---
zoom: 0.95
---

# From Earlier Lectures to Today

<div class="grid grid-cols-[4fr_3fr] gap-12">
<div>

### We already know
* Forward → loss → backward → update<br> *(Lecture 2)*
* Learning-rate search and weight decay<br> *(Lecture 2)*
* Backpropagation and initialization<br> *(Lecture 3)*
* Batch normalization and residual connections<br> *(Lecture 5)*
* Gradient clipping<br> *(Lecture 6)*

</div>
<div>

### Plan for today
1. **The objective**<br>training loss, new-data performance, loss landscapes

2. **Stochastic gradients**<br>noise, batch size, computation

3. **Better update rules**<br>momentum, adaptive rates, Adam(W)

4. **Training practice**<br>schedules, choice, debugging

</div>
</div>

<br>

#### The emerging question: **where does the next parameter update come from?**

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
