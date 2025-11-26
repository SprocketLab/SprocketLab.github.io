---
title: 'Activation Steering vs. Finetuning: How Different Are They Really?'
date: 2025-11-26
permalink: /posts/2025/11/actvweight/
tags:
  - Inference-time steering
  - Activation steering
  - Finetuning
  - Model adaptation
authors: "<a href='https://dyahadila.github.io/'>Dyah Adila</a>, John Cooper" 
thumbnail: /images/blogposts/actvweight/weight_v_act.svg
---

<script type="text/javascript" id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>
<script>
MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']]
  }
};
</script>

### The Paradigm: Activation Steering vs. Finetuning
Activation steering has emerged as an alternative to parameter-efficient finetuning (PEFT). Instead of updating model weights, steering directly adjusts intermediate activations at inference time, drastically reducing the number of trainable parameters. For example, ReFT [1] can match LoRA-level performance while using 15×–65× fewer parameters. Existing steering methods mainly differ in where they apply these interventions: ReFT modifies MLP outputs, LoFIT [2] steers at attention heads, and JoLA [3] jointly learns both the steering vectors and the intervention locations.

<p align="center">
<img src="https://sprocketlab.github.io/images/blogposts/actvweight/weight_v_act.svg">
</p>

### The Problem: We Are Steering in the Dark

Despite empirical success, we still lack a clear understanding of why steering works and how to reason about where is the best steering location. Two key questions remain open:

1. *The Location Dilemma.* Why do some steering locations outperform others? And does understanding how activation steering approximates weight updates help reveal which modules are best to steer? (Spoiler: it does)
2. *The Expressivity Gap.* SFT and LoRA leverage the model’s nonlinear transformations, while most steering methods rely on linear adapters. How much does this limit their ability to mimic the effects of weight updates?

This work investigates both questions.

**First: Where should we steer?**
We begin with a simple, linearized analysis of the module's output changes under weight updates compared to activation steering. This shows is that steering at different locations allows us to match the behavior of certain weight updates but not others. This gives us a principled way to identify which intervention points can, in principle, match weight updates. However, we notice that linear steering at either of these locations cannot completely capture the full behavior of weight updates

**Then: How expressive can steering really be?**
We then experiment with oracle steering, which, while not a practical method, provides a principled way to test which locations are best to steer at. With this tool, one pattern stands out: *the most expressive intervention point is the block output, after the skip connection*. Steering here can draw on both the skip-connection input and the transformed MLP output, instead of relying solely on either the MLP or attention pathway.

Motivated by this, we introduce a new activation adapter placed at each block output. It retains a LoRA-like low-rank structure but incorporates a nonlinearity after the down-projection. This allows it to capture some of the nonlinear effects characteristic of SFT, giving activation steering a more expressive update space.

**And finally: A bit of theory.**
No matter what the steering adapter is, if the adapter is able enough to match the fine-tuned model at each layer, the steered model will be able to match the fine-tuned model. The question we ask here: how accurate must we be to match a fine-tuned model closely?

We also show that, at least in some settings relating to the geometry of these hidden states and the residuals at each module, post-block steering can replicate a post-MLP steering. Also, we show that under some parameter settings, post-MLP cannot learn anything, while post-block steering still can.

## Some notation/background

Throughout this article, we will be looking at a number of different places to steer, along with different ways that we can steer. Even if some of these choices don't make sense at the moment, don't worry! A lot of this will be explained much more throughout this article. Use this section as a reference for any unclear notation/names as you read.

First, a Transformer model is made transformer blocks. Each block contains an Attention module and an MLP module. Unless otherwise specified, the MLP modules will be specifically GLU layers, a popular variant of standard 1-layer MLPs. The inputs to each submodule of each layer will pass through a LayerNorm. Each layer will involve two skip-connections, one around each submodule. This all can be seen in the picture below. Everything thus far is standard nomenclature of standard transformer architectures.

<p align="center">
<img src="https://sprocketlab.github.io/images/blogposts/actvweight/base-model.svg">
</p>

As for steering, there are 3 main variants we consider. First, pre-MLP steering involves steering attention outputs; most commonly done by steering the output of individual attention heads, *before* skip-connection and normalization. Second, post-MLP steering involves steering the output of the MLP/GLU layer *before* it goes through the skip-connection. Last, post-block steering involves steering the output of each block, which can be seen as equivalently steering the output of the MLP/GLU layer *after* it goes through the skip connection. In our notation, a GLU is represented as 

$$y_{\mathrm{GLU}}(h) = W_d(\sigma(W_g h) \odot W_u h)$$

The matrices $W_d, W_g, W_u$ are called the down-projection, gated, and ungated matrices respectively. When convenient, we will also write

$$ y(h)=W_d m(h), \quad m(h) = \sigma(a_g) \odot a_u, \quad a_g = W_g h, \quad a_u = W_u h $$

For mathematical notation, the hidden state will be represented as a vector $h$ and steering will be represented as $\delta h$. So, steering works by replacing $h$ with $h + \delta h$. Note that $\delta h$, the steering vector, can depend on the input. Sometimes this is written explicitly, but other times it is ommited.

**Fixed-vector:** The simpliest form of steering is by adding a fixed vector $v$ to the hidden state:

$$\delta h = v$$

**Linear/ReFT:** Linear steering involves some matrix $A$ and bias vector $b$, where the steering vector is a linear function of the hidden state. However, the matrix $A$ is usually replaced by a low-rank matrix (usually written as $W_1W_2^\top$ or $AB^\top$). The rank of this matrix is written as $r$ when necessary:

$$\delta h(h) = W_1W_2^\top h + b$$

**Non-linear:** This steering is parameterized by a 1-layer MLP with matrices $W_d, W_u$ as the down- and up-projections with SiLU activation $\phi$:

$$\delta h(h) = W_d\phi(W_u h)$$

**Feeely parametrized/Oracle:** In this case, the steering vector has no explicit parameterization. It can depend on the hidden state $h$ in any way. The oracle specifically will be given by the difference between the hidden states of the base and fine-tuned model

$$\delta h_{\mathrm{oracle}} = h_{\mathrm{FT}} - h_{\mathrm{base}}$$
 
With notation in place, we are now ready to begin our analysis!

<p align="center">
<img src="https://sprocketlab.github.io/images/blogposts/actvweight/stay-calm-panic.gif" width="500">
</p>

## Where should we steer?

Let's start with the main question that drives the rest of our analysis: 

> **At which points in the network can steering match the effect of updating the weights in that same module?**

We will examine pre-MLP (steering attention outputs, like LoFIT) and post-MLP (steering the MLP output before the skip connection, like ReFT) steering as they are the most common choice in the literature. These spots nicely *sandwich* the MLP, so the most immediate module affected by steering at these points is the MLP itself. Our first step is simple: compare the output changes caused by steering at these locations to the output changes caused by tuning the MLP weights.

*Note:* Before we start, it will feel like there is a lot of math here, but we promise, everything in this section is linear algebra.

<a href="https://imgflip.com/i/ad3bvp">
  <img src="https://i.imgflip.com/ad3bvp.jpg" width="500"/>
</a>

### Fine-tuning
Let the MLP output be

$$ y(h)=W_d m(h), \quad m(h) = \sigma(a_g) \odot a_u, \quad a_g = W_g h, \quad a_u = W_u h $$

Fine-tuning the weights gives us

$$ W_g \mapsto W_g + \delta W_g, \quad W_u \mapsto W_u + \delta W_u, \quad W_d \mapsto W_d + \delta W_d. $$

The updates $\delta W_g$ and $\delta W_u$ induce the following changes in their immediate outputs:

$$ \delta a_g = (\delta W_g) h, \quad \delta a_u = (\delta W_u) h.$$ 

A first order Taylor expansion of $m = \sigma(a_g) \odot a_u$ gives

$$ \delta m = (\sigma'(a_g) \odot a_u) \odot \delta a_g + \sigma(a_g) \odot \delta a_u + \text{(higher order terms)}.$$

Plugging in $\delta m$ into fine-tuning output gives us

$$ y_{\mathrm{FT}} (h) = (W_d + \delta W_d)(m+\delta m) \approx W_d m + \delta W_d m + W_d \delta m. $$

This yields the first-order shift caused by finetuning:

$$\boxed{\Delta y_{\mathrm{FT}} \equiv y_{\mathrm{FT}}(h) - y(h) \approx (\delta W_d) m + W_d [ (\sigma'(a_g) \odot a_u) \odot ((\delta W_g) h) + \sigma(a_g) \odot ((\delta W_u) h) ].}$$

Plugging $\delta m$ into $y = W_d m$ gives us

$$\boxed{
\Delta y_{\mathrm{pre}} \approx W_d [(\sigma'(a_g) \odot a_u) \odot (W_g \delta h) + \sigma (a_g) \odot (W_u \delta h)].
}$$


### Let's compare the finetuning and pre-MLP steering output shifts side by side

$$
\Delta y_{\mathrm{FT}} \approx (\delta W_d) m + W_d [ (\sigma'(a_g) \odot a_u) \odot ((\delta W_g) h) + \sigma(a_g) \odot ((\delta W_u) h) \\
\Delta y_{\mathrm{pre}} \approx W_d [(\sigma'(a_g) \odot a_u) \odot (W_g \delta h) + \sigma (a_g) \odot (W_u \delta h)].
$$

Notice that **the 2nd term in $\Delta y_{\mathrm{FT}}$ is structurally similar to $\Delta y_{\mathrm{pre}}$**. What does this imply?
> In principle, pre-MLP steering can match the shift caused by the updates $\delta W_u$ and $\delta W_g$, **if** there exist a $\delta h$ such that $W_g \delta h \approx (\delta W_g) h$ and $W_u \delta h \approx (\delta W_u) h$. 

Wait, what about the first term $(\delta W_d) m$? For a $\delta h$ to match this term, $(\delta W_d) m$ must lie in a space reachable by pre-MLP steering. Let's factor out $\delta h$ to see what that space looks like:

\begin{align*}
& \Delta y_{\mathrm{pre}} \approx W_d [(\sigma'(a_g) \odot a_u) W_g + \sigma(a_g) W_u] \delta h.
\end{align*}

Define $J(h) = [(\sigma'(a_g) \odot a_u) W_g + \sigma(a_g) W_u] \delta h \quad$ and $\quad A(h) = W_d J(h)$

We can rewrite $$\Delta y_{\mathrm{pre}} = A(h) \delta h.$$

For pre-MLP steering to match fine-tuning MLP shift, we must have: $$(\delta W_d) m \in \text{col}(A(h)).$$ What does this mean?

- Since $A(h) = W_d J(h) \subseteq \text{col}(W_d)$, any part of $(\delta W_d) m$ orthogonal to $\text{col}(W_d)$ is unreachable.
- If finetuning pushes $\delta W_d$ to new directions not spanned by $A(h)$, pre-MLP steering cannot match it.
- Note that as this condition must hold for *every token $h$*, this becomes really restrictive.

### Bottom line:
> Pre-MLP steering can partially imitate MLP finetuning (the $\delta W_g$ and $\delta W_u$ effects), but matching the full update is generally very hard, if not impossible.

### Post-MLP steering
Post-MLP steering directly modifies the **output of the MLP** $y$.  
Because it acts *after* all the nonlinearities inside the MLP, a sufficiently expressive parameterization could, in principle, reproduce **any** change made by fine-tuning the MLP weights. For example, if we were allowed a fully flexible oracle vector,

$$
\delta y = y_{\mathrm{FT}} - y,
$$

then adding this vector would give us the exact fine-tuned model output.

> This already puts post-MLP steering in a much better position than pre-MLP steering when it comes to matching MLP weight updates. **So are we all set with post-MLP steering as the way to go? Not quite.**

### **The missing piece: the skip connection**

Let’s look back at the structure of a Transformer block:

<p align="center">
<img src="https://sprocketlab.github.io/images/blogposts/actvweight/steering-locations.svg">
</p>

$$
\text{TransformerLayer}(\text{I}) = \text{I}' + \underbrace{\text{MLP}(\text{Norm}^{\text{MLP}}(\text{I}'))}_{\text{covered by post-MLP}}, \\
\text{I}' = \text{I} + \text{MHA}(\text{Norm}^{\text{MHA}}(\text{I}))
$$

Post-MLP steering only modifies the **MLP term**.  
But the block output is the **sum** of:

1. the MLP contribution (which we *can* steer), and  
2. the skip-connected input \(I'\) (which remains **unchanged** under post-MLP steering).

So even if post-MLP steering perfectly matches the MLP shift, it **cannot modify the skip-connection term**, which is half of the block output.


### **How big is this mismatch?**

<p align="center">
<img src="https://sprocketlab.github.io/images/blogposts/actvweight/norm.svg">
</p>

Across layers, post-MLP steering covers **at most ~70%** of the block output that fine-tuning changes, and in some layers, as little as **~40%**.  
The rest remains untouched, meaning post-MLP steering cannot fully replicate the effect of fine-tuning at that block.

---

### **Summary**

- Post-MLP steering is **more expressive** than pre-MLP steering because it can match arbitrary MLP updates.  
- **However**, it still misses all the changes happening through the skip-connection.  
- Matching MLP behavior is not enough, we also need a way to account for the skip path.


Now that we've sorted out “where to steer” part of the story, the next piece of the puzzle is how much steering can actually do. In other words: how expressive can activation steering be? Since ReFT is the most widely used steering method today, and represents the strongest linear steering baseline, it’s the right place to begin.

ReFT do a post-MLP steering and parameterize it as

$$
y_{\mathrm{ReFT}} = y + \textbf{R}^\top (\textbf{W} y + \textbf{b} - \textbf{R}y).
$$

$\textbf{R}, \textbf{W}, \textbf{b}$ are trainable parameters, where $\textbf{R} \in \mathbb{R}^{r \times d_{\mathrm{model}}}$ 
has a rank $r$ and orthonormal rows, 
$\textbf{W} \in \mathbb{R}^{r \times d_{\mathrm{model}}}$, and $\mathbf{b} \in \mathbb{R}^r$. 
The output shift caused by ReFT is:

$$
& \Delta y_{\mathrm{ReFT}} = y_{\mathrm{ReFT}} - y = (\textbf{R}^\top \textbf{W} - \textbf{R}^\top\textbf{R})y + \textbf{R}^\top b \\
& =  (\textbf{R}^\top \textbf{W} - \textbf{R}^\top\textbf{R})W_d m + \textbf{R}^\top b \\
& = \delta W_d ^{\mathrm{eff}} m + \textbf{R}^\top \textbf{b}, \quad \quad \delta W_d ^{\mathrm{eff}} =  (\textbf{R}^\top \textbf{W} - \textbf{R}^\top\textbf{R})W_d
$$

Now, let's compare $\Delta y_{\mathrm{ReFT}}$ with the $\Delta y_{\mathrm{FT}}$ we have from before. ReFT can induce a $\delta W_d$-like update, but only within the subspace spanned by $\textbf{R}$. So its ability to mimic full finetuning depends on the nature of $\delta W_d$ update, whether it is low-rank enough to fit inside that subspace. 

The second term ($\textbf{R}^\top\textbf{b}$) can only reproduce $\Delta y_{\mathrm{FT}}$'s $\delta W_u$ and $\delta W_g$ induced shift if it is approximately a linear function of the post-MLP output. This depends on how locally linear the mapping $h \mapsto y$ is.

When these conditions hold, ReFT can approximate the effects of MLP weight updates reasonably well. However, as we discussed earlier, post-MLP edits still leave a significant portion of the block’s computation untouched.

## How expressive can steering really be

Now that we've seen that steering after the skip-connection provides us with the largest expressivity for steering, let's see how good it really can be! In fact, our goal will be to match SFT, so let's see how far we get.

The simplest (and strongest) thing when given the SFT model would be to let the steering vectors be the oracle from above. Just to recall,

$$\delta h_{\mathrm{oracle}} = h_{\mathrm{FT}} - h_{\mathrm{base}}$$

so, quite literally,

$$h_{\mathrm{steer}} = h_{\mathrm{base}} + \delta h_{\mathrm{oracle}} = h_{\mathrm{FT}}$$

<p align="center">
<img src="https://sprocketlab.github.io/images/blogposts/actvweight/post-block-oracle.svg">
</p>


Okay, that's a bit much. This is simply overwritting the hidden state of the base model with the hidden state of the fine-tuned model (see the picture below). However, this still provided a lot of insight into where to steer, and the properties of a desired steering method.

Taking all of these different oracle steering vectors, their properties can be looked at for patterns to exploit. We quickly found that these vectors were close to low-rank (their covariance had a concentrated spectrum). But be careful! Just because the oracle steering vectors all almost exist in some low-dimension subspace, it does not mean the transformation from hidden states to steering vectors is linear! Sometimes it might be, sometimes is won't. 

In fact, if we try and replace the oracle with the best linear approximation of the map between hidden states and steering vectors, we find that the oracle goes from perfect matching to similar-to-or-worse-than ReFT!

<p align="center">
<img src="https://sprocketlab.github.io/images/blogposts/actvweight/llama_sequential.png">
</p>


Here, we are taking generations on some prompt and comparing the KL divergence between the fully fine-tunded model's output probabilities and those of the other models. The oracle often deviates further from the fine-tuned model than ReFT does, although not by much.

The best way around this would be to learn a *low-rank, non-linear function* as the map between the hidden states and the steering vectors. What better than a small autoencoder! Its output space would be constrained by the column space of the up-projection, so it will still be low-rank. 

At this point though, it seemed like we understood what would likely work as a steering vector, so we moved away from using the oracle as the gold-standard to match and now worked on training these adapters end-to-end.

### Post-Block Performance

Now, we began learning this steering vectors *without* a guide. Low-rank steering modules were added at the end of each block and trained, similar to LoRA/PEFT except on a module never present in training. Following the above discussion, two major variants were tested: linear and non-linear. In the linear case, the steering was done by a low-rank matrix. In the non-linear case, a non-linearity was placed between the down- and up-projection, making this an autoencoder. This was inspired by possibly needed a non-linearity from above, but we still want to preserve that low-rank structure of the steering vectors.

Here are the results we're currently seeing for 1B-parameter models:

