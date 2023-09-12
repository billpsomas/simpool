# Keep It SimPool: Who Said Supervised Transformers Suffer from Attention Deficit?
PyTorch implementation and pretrained models for SimPool. 

<div align="center">
  <img width="100%" alt="SimPool illustration" src=".github/overview.png">
</div>

## Overview
Convolutional networks and vision transformers have different forms of pairwise interactions, pooling across layers and *pooling at the end of the network*. Does the latter really need to be different :grey_question:

As a by-product of pooling, vision transformers provide spatial attention for free, but this is most often of *low quality* unless self-supervised, which is not well studied. Is supervision really the problem?

In this work, we develop a *generic pooling framework* and then we formulate a number of existing methods as instantiations. By discussing the properties of each group of methods, we derive *SimPool*, a simple attention-based pooling mechanism as a replacement of the default one for both convolutional and transformer encoders. We find that, whether *supervised* or *self-supervised*, this improves performance on pre-training and downstream tasks and provides attention maps *delineating object boundaries* in all cases.
One could thus call SimPool *universal*. To our knowledge, we are the first to obtain attention maps in supervised transformers of at least as good quality as self-supervised, without explicit losses or modifying the architecture.

<div align="center">
  <img width="100%" alt="SimPool attention maps" src=".github/attmaps.png">
</div>

We introduce SimPool, a simple attention-based pooling method at the end of network, obtaining clean attention maps under supervision or self-supervision. Attention maps of ViT-S trained on ImageNet-1k. For baseline, we use the mean attention map of the [CLS] token. For SimPool, we use the attention map a. Note that when using SimPool with Vision Transformers, the [CLS] token is *completely discarded*. 

> :loudspeaker: **NOTE: Considering integrating SimPool into your workflow?**  
> Use SimPool when you need attention maps of the highest quality. Dive into our README to learn more!

