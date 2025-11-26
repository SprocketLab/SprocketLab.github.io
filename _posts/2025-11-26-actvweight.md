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
<img src="https://sprocketlab.github.io/images/blogposts/actvweight/stay-calm-panic.gif" width="200">
</p>