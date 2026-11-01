---
layout: section
---

# Quiz: Optimization Algorithms

Five questions — choose first, then reveal the answer.

<!--
Allow about 2 minutes per question. Ask for a show of hands or use the course polling tool. The explanations are
more important than the letter answers.
-->

---
zoom: 0.9
---

# Quiz 1: Batch Size and Gradient Noise

You increase the mini-batch from **16 to 256** examples, keeping the same model and learning rate. In the ideal independent-example model, what happens?

A. Gradient-noise standard deviation falls **4×**; steps per epoch fall **16×**  
B. Gradient-noise standard deviation falls **16×**; steps per epoch fall **4×**  
C. Gradient-noise standard deviation falls **2×**; steps per epoch stay fixed  
D. Both noise and steps per epoch stay fixed

<v-click>

### Answer: **A.**
Noise scales as $1/\sqrt b$: $\sqrt{256/16}=4$. The batch contains $16\times$ more examples, so there are $16\times$ fewer updates per epoch.

</v-click>

<!--
Answer A. This checks two separate consequences of batch size: statistical noise and update frequency. It does not
claim that the larger batch trains better; the learning rate and wall-clock comparison still matter.
-->

---
zoom: 0.9
---

# Quiz 2: Momentum Arithmetic

Using $v_t=0.5v_{t-1}+g_t$ with $v_0=0$, suppose $g_1=g_2=g_3=2$. What is $v_3$?

A. 2  
B. 3  
C. **3.5**  
D. 6

<v-click>

### Answer: **C. 3.5**
$v_1=2$, $v_2=0.5\cdot2+2=3$, and $v_3=0.5\cdot3+2=3.5$. Past gradients are downweighted, not simply added forever.

</v-click>

<!--
Answer C. If someone chooses D, they are treating momentum as an unnormalized sum without decay. If someone uses
0.5 as the current gradient's coefficient, remind them of the convention shown on the momentum slide.
-->

---
zoom: 0.9
---

# Quiz 3: Adam's First Step

At $t=1$, a coordinate has $g_1=2$. With $\beta_1=0.9$ and $\beta_2=0.999$, starting from zero, what are the **bias-corrected** first and second moments?

A. $\hat m_1=0.2,\ \hat s_1=0.004$  
B. $\hat m_1=2,\ \hat s_1=4$  
C. $\hat m_1=1.8,\ \hat s_1=3.996$  
D. $\hat m_1=0,\ \hat s_1=0$

<v-click>

### Answer: **B.**
The raw values are $m_1=0.1\cdot2=0.2$ and $s_1=0.001\cdot4=0.004$. Divide by $1-\beta_1=0.1$ and $1-\beta_2=0.001$: $\hat m_1=2,\ \hat s_1=4$.

</v-click>

<!--
Answer B. Choice A gives the uncorrected moments. Bias correction removes the zero-initialization bias; it does not
make the gradient itself larger in later steps.
-->

---
zoom: 0.9
---

# Quiz 4: AdamW and Weight Decay

Why is AdamW with $\lambda>0$ different from adding an L2 penalty to Adam's loss?

A. AdamW uses no learning rate  
B. AdamW applies weight decay **outside** the adaptive moment scaling  
C. AdamW removes the second-moment estimate  
D. There is no difference: both always produce identical updates

<v-click>

### Answer: **B.**
AdamW applies $\lambda\theta$ directly in the parameter update. In Adam + L2, the penalty enters $g_t$ and is scaled by $\sqrt{\hat s_t}$ along with everything else.

</v-click>

<!--
Answer B. Review Lecture 2 for weight decay as shrinkage. AdamW still has a learning rate and both moment estimates;
it changes where the decay is applied, not the rest of Adam.
-->

---
zoom: 0.9
---

# Quiz 5: Debugging a Diverging Run

You double the learning rate. A few hundred steps later, the loss oscillates and becomes **NaN**. What is the best first response?

A. Double the batch size and keep going  
B. Switch to AdamW with the same large learning rate  
C. Return to the stable learning rate; inspect gradient norms and clip spikes if needed  
D. Add weight decay and train for twice as long

<v-click>

### Answer: **C.**
First restore the known-stable setting. Then inspect gradients and data for instability. Gradient clipping can limit spikes *(Lecture 6)*; changing the optimizer does not make an excessive learning rate safe.

</v-click>

<!--
Answer C. A larger batch can reduce noise but is not the first fix for a change that immediately destabilized training.
Weight decay is for regularization, not an emergency cure for numerical divergence.
-->
