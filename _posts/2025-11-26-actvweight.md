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