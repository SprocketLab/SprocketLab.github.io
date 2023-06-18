---
title: 'Aggregating Foundation Model Objects'
date: 2023-06-15
permalink: /posts/2023/06/lifting-ws/
tags:
  - Language Models
  - Weak Supervision
  - Structured Prediction
  - Tensor Decomposition
  - Non-Euclidean ML
excerpt: ''
authors: "<a href=https://harit7.github.io/'>Harit Vishwakarma</a> and <a href='https://pages.cs.wisc.edu/~fredsala/'> Fred Sala </a>" 
---


One exciting aspect of large pretrained 'foundation' models is that it is easy to obtain *multiple observations* by repeatedly querying. The most straightforward example is to obtain multiple answers to a question by varying the prompt, as shown below. Naturally, we'd like to **aggregate** these outputs in such a way that we obtain a better estimate of the ground truth than any single answer on its own. How can we do this? Ideally, we'd like to 
* Take into account that some estimated objects are closer to the ground truth than others, i.e., are more accurate, and upweight these, 
* Be fully unsupervised---so that we have no access to the ground truth and can be fully zero-shot, 
* Be able to aggregate structured objects---not just model outputs, but chains, trees, and other intermediate structures used in techniques like [chain-of-thought](https://arxiv.org/abs/2201.11903) (CoT) prompting and other reasoning approaches. 

In this post we discuss a simple way to do this based on one of our [NeurIPS '22 papers](https://proceedings.neurips.cc/paper_files/paper/2022/file/f463d31ed2fdd7b0ec585c041ec1baa8-Paper-Conference.pdf). The core principle is a (very general) form of the weak supervision algorithms that we've been [playing with](http://ai.stanford.edu/blog/weak-supervision/) [for several](https://hazyresearch.stanford.edu/blog/2020-02-28-flyingsquid) years. For binary outputs, this idea has already been successfully used in our [Ask Me Anything prompting strategy](https://arxiv.org/abs/2210.02441). Here, we focus on lifting this to the richer structures needed for CoT and other techniques.

![ Weak Supervision to aggregate LLM outputs fig:ws-llm-agg}](/images/blogposts/lifting-ws/ws-ama-llm-agg.jpg "Weak Supervision to aggregate LLM outputs ")

Warning: our discussion will get a bit technical---but we promise it will be fun! In fact we'll get to connect to a ton of different fields, including graphical models, unsupervised learning, embeddings and non-Euclidean geometry, tensor algorithms, and more! 

First, our roadmap. We will
* Cover some well-tread ground on the fundamentals of simple aggregation. We'll model noisy observations of binary objects and describe a very simple approach to learn how accurate the observations are, without ground truth. This part will also serve as a short introduction to weak supervision.
* Apply a more powerful algorithm for the above based on tensor decomposition---enabling us to relax our modeling assumptions for aggregation.
* Figure out how to scale it up to rich structures by operating on a very special type of embedding, called pseudo-Euclidean.
* Show how this can help us improve CoT beyond simple approaches like majority vote. 

# Aggregation Fundamentals
Let's take the example in the figure above. We are performing a basic email classification task, where we want to categorize each message as spam or not. We repeatedly query the model by varying the prompt, obtaining a number of observations for each email.

Borrowing from the language of weak supervision, we'll refer to each prompting approach as a labeling functions. These labeling functions are just estimates of the ground truth answer for whatever task we're interested in. What can we do with these? First, let's collect the outputs. These are arranged in a matrix as shown in figure below. The instances (examples) are the emails. Of course, the column for the ground truth label $Y$ is just a placeholder since we don't get to observe these. 

![ This how the data points and LF outputs usually look. \label{fig:lf-outputs}](/images/blogposts/lifting-ws/ws-example-table.png "Data points and LF outputs.")

After observing the outputs of labeling functions, **the goal of aggregeation is to estimate the ground truth label---and hopefully more accurately than any source (labeling function) by itself**! A naive but reasonable way to aggregate is to take the *majority vote* of the outputs for each point. This approach will work well when the LFs are independent and have similar qualities. However, some LFs could be more accurate and some more noisy. They might also be correlated. This can make majority vote less effective. 

How can we model these possibilities? [Weak supervision approaches](https://dawn.cs.stanford.edu/pubs/snorkel-nips2016.pdf) often take the joint distribution of $Y,\lambda_1, \ldots \lambda_m$  as a probabilistic graphical model, such as the Ising model:

 $$ P_{\theta}(\lambda_1,\lambda_2,\ldots \lambda_m,Y) = \frac{1}{Z}\exp \Big( \theta_Y Y + \sum_{i=1}^m \theta_i \lambda_i Y + \sum_{(i,j)\in E} \theta_{ij}\lambda_i \lambda_j \Big) $$

What does this do for us? First, we can now think of learning the accuracies and correlations described above as learning the parameters of this model. Note that unlike conventional learning for graphical models, we have a *latent* variable problem, as we do not observe $Y$. If we have learned these parameters, we can rely on the estimated model to infer the true labels. The resulting pipeline looks like.
![Standard weak supervision pipeline \label{fig:lf-outputs}](/images/blogposts/lifting-ws/ws-pipeline.png "Standard weak supervision pipeline")

The $\theta$ parameters above encode how accurate each of the labeling functions are, with a large $\theta_i$ indicating that the $i$th noisy estimate frequently agrees with $Y$, the ground truth. How do we estimate these? We'll need a few technical pieces from the graphical model literature. It turns out that we need only estimate the *mean parameters*---terms like $\mathbb{E}[\lambda_i Y]$ and  $\mathbb{E}[\lambda_i \lambda_j]$! The correlation terms  $\mathbb{E}[\lambda_i \lambda_j]$ do not involve $Y$. How can we learn these? As long as we know the structure (the edge set E), the rest is easy, since these terms are observed. 

How about the accuracy parameters i.e., the correlations between $\lambda_i$ and $Y$ ? This is challenging as we don't get to see any ground truth! There are classical methods like [EM (Expectation-Maximization)](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm) and variants such as [David-Skene](http://crowdsourcing-class.org/readings/downloads/ml/EM.pdf) that could be applied. However, these approaches are prone to converging to local optima and sometimes perform poorly. A simple and elegant approach, [Flying Squid](https://hazyresearch.stanford.edu/blog/2020-02-28-flyingsquid), based on the [Method of Moments](https://cs.stanford.edu/~pliang/papers/graphical-icml2014.pdf), to the rescue!
The key idea is based on the observation that for any three  conditionally independent LFs, $\lambda_1,\lambda_2,\lambda_3$  the second order moments with binary labels can be written as,

$$ \mathbb{E}[\lambda_1\lambda_2] = \mathbb{E}[\lambda_1 Y]\mathbb{E}[\lambda_2 Y]$$

$$ \mathbb{E}[\lambda_2\lambda_3] = \mathbb{E}[\lambda_2 Y]\mathbb{E}[\lambda_3 Y]$$

$$ \mathbb{E}[\lambda_3\lambda_1] = \mathbb{E}[\lambda_3 Y]\mathbb{E}[\lambda_1 Y]$$

This system of three equations can be solved analytically for $\mathbb{E}[\lambda_i Y]$. 
$$|\mathbb{E}[\lambda_1 Y]| = \sqrt{\frac{\mathbb{E}[\lambda_1\lambda_2] \mathbb{E}[\lambda_3\lambda_1] }{\mathbb{E}[\lambda_2\lambda_3]}}, |\mathbb{E}[\lambda_2 Y] |= \sqrt{\frac{\mathbb{E}[\lambda_1\lambda_2] \mathbb{E}[\lambda_2\lambda_3] }{\mathbb{E}[\lambda_3\lambda_1]}}, |\mathbb{E}[\lambda_3 Y]| = \sqrt{\frac{\mathbb{E}[\lambda_2\lambda_3] \mathbb{E}[\lambda_3\lambda_1] }{\mathbb{E}[\lambda_1\lambda_2]}}$$ 
This analytical solution is easy to obtain for the binary classification setting. All that is left is to figure out the signs of the above, in order to break symmetry. As long as our sources are better than random on average, this can be done. 

This basic idea can also be extended easily to multi-class settings by solving multiple one vs rest binary classification problems. This method enjoys nice theoretical guarantees and works pretty well for classification settings especially when the number of classes is small---and when the model has special kinds of symmetry. More details about Flying Squid can be found in the [blog post](https://hazyresearch.stanford.edu/blog/2020-02-28-flyingsquid) and [paper](https://arxiv.org/abs/2002.11955). 


# Stronger Aggregation with Tensor Decomposition
As we saw, the main challenge in WS is to estimate the accuracies $\theta_i$ of the labeling functions without having access to the ground truth. While approaches like FlyingSquid are simple and efficient, they make pretty strong assumptions. If we want to handle outputs that have high-cardinality or special structure, we may need a more powerful tool.

[Tensor decompositions](https://arxiv.org/abs/1408.0553) are a great candidate for this---having already been used for solving latent variable problems. Before we proceed, let's see how we can adapt this class of algorithms to our aggregation setting. 

We will first discuss the classical multi-view mixture model learning using tensor decomposition and see that it works on-par with existing methods like Flying Squid when there are small number of classes and we use 1-hot encodings to represent them. We will see that this approach doesn't scale well when the labels live higher-cardinality scenarios with structure. To make tensor decomposition work in such general settings we first embed the labels in a special geometric space called pseudo-Euclidean and then perform tensor decomposition on these vectors. 

##  Multi-View Mixture and Tensor Decomposition
We can think of the observed labeling functions outputs as the observations from a multi-view mixture model i.e., each LF $\lambda_a$ is a *view* of the true label $Y$. In a multi-view mixture model, multiple views $$\{\lambda_{a}\}_{a=1}^m$$ of a latent variable $Y$ are observed. These views are independent when conditioned on $Y$. 
i.e. $\lambda_{a}|Y=y$ is conditionally independent of $\lambda_{b}|Y=y$ for all $a,b$ . This mixture model is depicted as a graphical model in the figure below.
<img width="300" style="float:right" src="/images/blogposts/lifting-ws/multi-view-mixture-fig.png " />

Now, suppose we have a cardinality $k$ problem (the true label $Y$ takes $k$ values). We use one-hot vector representations of the labels ( denoted in bold-face ). Let $$\mathbb{E}[{\mathbf{\lambda}}_a|Y=y] = {\mathbf{\mu}}_{ay}$$ denote the mean of $\mathbf{\lambda}_a$ conditioned on the true label $y$ (for all $a$ and $y$). Then it is easy to see the following for the tensor product (third order moment)  
 of any three conditionally independent ${\mathbf{\lambda}}_a,{\mathbf{\lambda}}_b,{\mathbf{\lambda}}_c$ ,

$$ {\mathbf{T}} = \mathbb{E}_{\lambda_a,\lambda_b,\lambda_c,y}[{\mathbf{\lambda}}_a \otimes {\mathbf{\lambda}}_b \otimes {\mathbf{\lambda}}_c] = \sum_{y\in[k]} w_y {\mathbf{\mu}}_{a,y} \otimes {\mathbf{\mu}}_{b,y} \otimes {\mathbf{\mu}}_{c,y} $$ 

 i.e. $\mathbf{T}$ can be written as a sum of $k$ rank-1 tensors or in other words $\mathbf{T}$ can be factorized into $k$ rank-1 tensors. Note that we do not know the true distribution of $\lambda,y$. Instead we have $n$ i.i.d observations 
$$\{ {\mathbf{\lambda}}_{a,i}\}_{a\in [m],i\in [n]}$$. Using these we can produce an empirical estimate of $\mathbf{T}$:

$$\hat{\mathbf{T}} = \hat{\mathbb{E}}[{\mathbf{\lambda}}_a \otimes {\mathbf{\lambda}}_b \otimes {\mathbf{\lambda}}_c] = \frac{1}{n}\sum_{i\in[n]}  {\mathbf{\lambda}}_{a,i} \otimes {\mathbf{\lambda}}_{b,i} \otimes {\mathbf{\lambda}}_{c,i}$$

Suppose $$\tilde{\mathbf{T}} = \sum_{y\in[k]}\hat{w}_y \hat{\mathbf{\mu}}_{a,y}\otimes \hat{\mathbf{\mu}}_{b,y} \otimes\hat{\mathbf{\mu}}_{c,y}$$ is a rank-k factorization of the empirical tensor $\hat{\mathbf{T}}$. If $\hat{\mathbf{T}}$ is a good approximation of the true tensor $\mathbf{T}$ and  $\tilde{\mathbf{T}}$ is a good approximation of $\hat{\mathbf{T}}$ then we have that $$\hat{\mathbf{\mu}}_{a,y}$$ is good approximation of the true mean parameters ${\mathbf{\mu}}_{a,y}$. 
These results are shown in [Anandkumar et al. 2012,](https://arxiv.org/abs/1210.7559)[ 2014]( https://arxiv.org/abs/1408.0553) under certain regularity conditions.

Using the estimates $\hat{\mathbf{\mu}}_{a,y}$ we obtain estimates of our canonical $\theta$ parameters and apply the inference procedure to infer the true labels based on the labeling functions outputs. We will refer to our weak supervision method based tensor decomposition as tensor label model.

## Tensor Label model is Competitive in Usual Settings
The big question---how well does this work? We run a simple experiment on simulated labeling functions to show that this method is a competitive baseline. For this we simulate three labeling functions with  $\theta=[4,0.5,0.5]$. We run the tensor label model on the 1-hot encodings of the labels and compare the accuracy of the inferred pseudolabels against the Flying Squid and majority vote baselines. The results are shown in figure \ref{fig:mean_acc_usual} ( averaged over 100 trials). As expected, the tensor label model has competitive performance but due to the use of 1-hot encodings---leading to high dimensionality---its performance also degrades when we increase the cardinality of the label space.


<!--
![ Mean accuracies of methods on multi-class classification with 1-hot encodings. \label{fig:mean_acc_usual} ](/images/blogposts/lifting-ws/figure_mean_acc_cg_all.jpg "Mean accuracies of methods on multi-class classification with 1-hot encodings"){width=250}
-->
This is indeed quite encouraging and motivates us to apply it beyond simple classification settings.

# Weak Supervision Beyond Classification 
As we alluded to, it is common for the labels to be much more diverse. Some of the examples we have looked at are movie rankings, dependency parse trees, classes of objects having some hierarchy etc. We can think of these label spaces as finite metric spaces---since they have natural distance functions---and seek to apply weak supervision. Our approach consists of two high level steps: first we learn isometric representations of the labels using a classical tool  [Pseudo-Euclidean Embeddings (PSE)](https://en.wikipedia.org/wiki/Pseudo-Euclidean_space)  and then adapt the above tensor label model to work with PSE embeddings. As we shall see, both of these steps are critical. 

![ Our Weak Superivision Pipeline ](/images/blogposts/lifting-ws/fig-ws-sp-paper.jpg "Our Weak Supervision Pipeline for Finite Metric Spaces")


## Distortion Free Embeddings
Working directly with the discrete metric spaces is challenging---we can't use our favorite off-the-shelf optimization approaches! To make life easy we'll do the usual: work with vector space representations of the objects! Learning such representations has been a very active area of research over several decades. Here we are particularly interested in learning *isometric* -- distance preserving embeddings. This is crucial: if we distort these distances, the pseudolabels coming out of weak supervision might be quite far from the true labels, precisely what we're trying to avoid!

We use a classical method called [Pseudo-Euclidean Embeddings (PSE)](https://en.wikipedia.org/wiki/Pseudo-Euclidean_space). It is itself a generalization of another classical method called [Multi-Dimensional Scaling(MDS)](https://en.wikipedia.org/wiki/Multidimensional_scaling). The main benefit of PSE over MDS is that it can isometrically embed metric spaces that cannot be isometrically embeddable in Euclidean space. 
   
To understand the utility of PSE better, consider the following examples of two graphs in the below figure and learning their node embeddings using MDS, PSE and 1-hot encoding. Clearly, 1-hot encoding is not a scalable solution as the embedding dimension will be the same as the number of nodes, which will quickly be prohibitively large. On the other hand, MDS gives low dimensional embeddings but cannot give isometric embeddings for general metric spaces. In the below example, the best the MDS can give is 2-D embeddings which have high distortion ( not isometric). 
   In contrast, PSE gives 3-D embeddings with nearly 0 distortion (isometric). For the tree example we can also see that the dimension of PSE doesn't increase with more nodes.


![ Examples of metric spaces and their embeddings using MDS and PSE  \label{fig:pse-examples} ](/images/blogposts/lifting-ws/pse-examples-2-trees.jpg "Examples of metric spaces and their embeddings using MDS and PSE")

A vector $\mathbf{u}$ in a pseudo-Euclidean space $\mathbb{R}^{d^+,d^-}$ has two parts: $\mathbf{u}^+ \in \mathbb{R}^{d^+}$ and ${\mathbf{u}}^- \in \mathbb{R}^{d^-}$. The dot product and the squared distance between any two vectors ${\mathbf{u}},{\mathbf{v}}$ are $\langle {\mathbf{u}}, {\mathbf{v}}\rangle_{\phi} = \langle {\mathbf{u}}^+,{\mathbf{v}}^+ \rangle - \langle {\mathbf{u}}^-,{\mathbf{v}}^- \rangle$ and $$d^2_{\phi}(\mathbf{u},\mathbf{v}) =\lVert \mathbf{u}^{+}-\mathbf{v}^{+}\rVert_2^2 - \lVert\mathbf{u}^{-}-\mathbf{v}^{-}\rVert_2^2$$.  These properties enable isometric embeddings: the distance can be decomposed into two components that are individually induced from p.s.d. inner products---and can thus be embedded via MDS. Indeed, pseudo-Euclidean embeddings effectively run MDS for each component. To recover the original distance, we obtain $\lVert \mathbf{u}^{+}-\mathbf{v}^{+}\rVert_2^2$ and $\lVert \mathbf{u}^{-}-\mathbf{v}^{-}\rVert_2^2$ and subtract. More details on pseudo-euclidean embeddings can be found in a comprehensive article, [A new approach to pattern recoginition by Lev Goldfarb ](https://www.researchgate.net/publication/233408916_A_new_approach_to_pattern_recognition).

## Upgrading Label Model with PSE
Armed with the powerful technique pseudo-euclidean embeddings (PSE) we get isometric representations of the points in the label space (potentially in lower dimensions) then represent the LF outputs $\lambda_{a,i}$ in the PSE space and solve the parameter estimation problem in this space using the tensor decomposition method discussed above. The original tensor decomposition algorithm was designed for points in Euclidean space so we cannot apply it off-the-shelf when the points are living in PSE spaces. We overcome this issue by using the fact that the two parts of any vector in PSE are individually in Euclidean spaces $\mathbb{R}^{d^+},\mathbb{R}^{d-}$ respectively. This allows us to treat the positive and negative components $$\mathbf{\lambda}_{a}^+ \in \mathbb{R}^{d^+}$$ and $$\mathbf{\lambda}_{a}^{-} \in \mathbb{R}^{d^-}$$ of our pseudo-Euclidean embedding as separate multi-view mixtures. We apply tensor decomposition on them separately, which gives us mean parameters $$\hat{\mathbf{\mu}}^+_{a,y}$$ and  $$\hat{\mathbf{\mu}}^-_{a,y}$$ for each $a,y$. Using these we obtain our estimates of the canonical parameters $$\hat{\mathbf{\theta}}$$.

With this adaptation, we retain the nice theoretical guarantees of tensor decomposition for parameter recovery while working with any finite metric space. We can also see the benefit of our approach on a simple synthetic data experiment on the tree metric we saw earlier. In this experiment, we simulate three labeling functions on the tree metric with three branches with $b$ number of nodes in each branch. We use $\theta=[4,0.5,0.5]$ i.e. first LF is highly accurate and the other two are somewhat noisy. We run two variations of our method one with PSE embeddings and the other with 1-hot embeddings of the labels. We keep the number of samples $n=1000$ fixed and vary the number nodes $b$ to increase the cardinality of the label space. The results can be seen in figure \ref{fig:mean_acc_pse}. As expected, using PSE embeddings we can achieve much better accuracy of the inferred pseudolabels and unlike other methods this performance does not degrade with higher cardinality as this metric space is isometrically embeddable in 3-dimensional PSE space.





![](/images/blogposts/lifting-ws/figure_mean_acc_cg_all.jpg)  |  ![](/images/blogposts/lifting-ws/figure_mean_acc_tree_all.png)
: ------------------------- : | : ------------------------- :
Performance with 0-1 metric   |   Performance on Tree Space


<!--- 
![  Mean accuracies of methods when the label space is the tree in figure \ref{fig:pse-examples} \label{fig:mean_acc_pse}](/images/blogposts/lifting-ws/figure_mean_acc_tree_all.png){width=250}
--->

# Chain of Thought Reasoning Application 
Our technique is quite general and can be applied in any setting where we can only get multiple noisy observations of a ground truth object from some discrete metric space and want to estimate/recover the ground truth using these noisy observations. 

To illustrate its practical application, let's consider the design of prompts with in-context examples for Large Language Models (LLMs). The in-context examples typically consist of paired input and output data, which effectively guide LLMs in comprehending the task at hand and generating accurate predictions. Recent advancements in this area have shed light on the effectiveness of prompts that incorporate explicit steps known as Chain of Thoughts (CoT). These step-by-step instructions facilitate LLMs in making precise predictions while providing detailed reasoning steps. Building upon this concept, more nuanced variations such as Tree of Thought (ToT) and Graph of Thought (GoT) have emerged. These expanded frameworks have demonstrated impressive efficacy when tackling complex reasoning problems with LLMs.


While highly effective, they require access to high quality explanations (CoT) which can be a bottleneck in broad applicability of these methods. Nevertheless, one can always come up with heuristics or have inexpensive sources that can provide potentially noisy reasoning steps. How can we utilize these to construct accurate chain or tree or graph of thoughts? Can we utilize the weak supervision (WS) principles to solve this problem?  

![ Expressions CoT. \label{fig:exp-cot}](/images/blogposts/lifting-ws/expressions-cot-2.jpg "Expressions CoT.")

<img width="300" style="float:right" src="/images/blogposts/lifting-ws/bar_plot_with_error_bars.png " />
 
Indeed, our method for WS on structured prediction can help here and we demonstrate its effectiveness with the help of an example for Chain of Thought reasoning. In particular, we consider the Game of 24 which is a complex reasoning puzzle with 4 numbers from 1 to 13 as input and the expected output is an expression using the given numbers and basic arithmetic operations (+,-,x,/) so that the expression evaluates to 24. Note that this task can be easily solved by enumerating all possible expressions and selecting the ones that evaluate to 24. However, we  are interested in solving this task using LLMs by providing it some in-context examples with chain of thought reasoning. Here the CoT steps could be an expression broken down into multiple steps. 
We use the same 1362 puzzles as in the [Tree of Thought paper](https://arxiv.org/abs/2305.10601)  and simulate 3 sources with different noise levels $\theta= [5,0.6,0.5]$ that can provide noisy expressions (CoTs). We then apply our procedure based on pseudo-euclidean embeddings and tensor decomposition to recover the true expressions and evaluate the recovered expressions for the correctness. We run this procedure 10 times with different random seeds and report the mean accuracies in the above bar chart. We can clearly see that our method based on tensor decompositions output performs a naive majority vote. 
 
 
Although on a small scale setup, these findings are quite exciting and demonstrate the potential of weak supervision in structured predictions settings such as CoT, ToT, GoT reasoning with LLMs.   

# Takeaways and Future Work
1. Weak supervision techniques can be super useful in aggregating noisy labeling sources to guess true labels.
2. Building on top of classical tools -- tensor decomposition and pseudo-Euclidean embeddings, we can build a weak supervision method that works well for any finite metric space and has favorable theoretical guarantees. 
3. Our current analysis is with isometric embeddings, it would be interesting to see how the guarantees change when there is some distortion allowed in the embeddings. We are interested in exploring novel structured prediction applications 

For more details, check out our paper on [arXiv](https://arxiv.org/abs/2211.13375), and our code on [Github](https://github.com/SprocketLab/WS-Structured-Prediction)!



