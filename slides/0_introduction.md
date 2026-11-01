---
zoom: 0.9
---

# Essentials of Artificial Neural Networks

### Building blocks:
<div class="grid grid-cols-[3fr_2fr_2fr] gap-3">
<div>

* Neuron
* Loss function
* Activation function
* <span style="color:#FA9370">**Optimizer**</span>
</div>

<div>

* Linear layer
* Convolution layer
* Pooling layer
* Recurrent layer
</div>

<div>
  <figure>
    <img src="/lego_A.jpg" style="width: 200px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute;"><br>Image source:
      <a href="http://sgaguilarmjargueso.blogspot.com/2014/08/de-lego.html">http://sgaguilarmjargueso.blogspot.com</a>
    </figcaption>
  </figure>
</div>
</div>

### Today's focus:
<div class="grid grid-cols-[2fr_3fr_3fr] gap-8">
<div>

* Noisy gradients
* Mini-batch SGD
* Momentum

</div>
<div>

* Adagrad, RMSProp
* Adam and AdamW
* Learning-rate schedules
</div>

<div>
<br>

* Choosing and debugging an optimizer
</div>
</div>

<span class="refs">Main text: [d2l.ai, Chapter 12: Optimization Algorithms](https://d2l.ai/chapter_optimization/index.html) · Refresher: [J. Starmer, StatQuest: Gradient Descent, Step-by-Step](https://www.youtube.com/watch?v=sDv4f4s2SB8)</span>

<!--
The architecture and the loss define the objective. The optimizer is the update rule that turns backpropagation's
gradient into new parameters. Today is about that last step, not about a new layer.

Prerequisite check: students already know the training loop (Lecture 2), backpropagation (Lecture 3), batch
normalization (Lecture 5), and gradient clipping (Lecture 6). Do not reteach them; use them as tools.
-->
