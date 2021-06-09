---
layout: post
title:  "Advanced GANs - Exploring Normalization Techniques for GAN training: Self-Attention and Spectral Norm"
date:   2018-08-11 08:00:00 -0300
categories: deep learning
short_description: "Lately, Generative Models are drawing a lot of attention. Much of that comes from Generative Adversarial Networks (GANs). Let's investigate some recent techniques for improving GAN training."
reference_file: advanced_gans_ref
image_url: "/assets/advanced_gans/cover.png"
tags: machine-learning deep-learning tensorflow keras python generative-models gans attention 
---

## Introduction

Lately, Generative Models are drawing a lot of attention. Much of that comes from Generative Adversarial Networks (GANs). Invented by [Goodfellow et al](#4), GANs are a framework in which two players compete with one another. The two actors, the generator G and discriminator D are both represented by function approximators. Moreover, they play different roles in the game.

Given a training data D<sub>train</sub>, the generator creates samples as an attempt to mimic the ones from the same probability distribution as D<sub>train</sub>.

The discriminator, on the other hand, is a common binary classifier. It has two main jobs. First, it categorizes whether its received input comes from the true data distribution (D<sub>train</sub>) or from the Generator distribution. In addition, D also guides G to create more realistic samples by passing to G its gradients. In fact, taking the gradients from D is the only way G optimize its parameters.

In this game, G takes random noise as input and generates a sample image G<sub>sample</sub>. This sample is designed to maximize the probability of making D mistakes it as coming from real training set D<sub>train</sub>.  

During training, D receives half of the time images from the training set D<sub>train</sub>, and the other half, images from the generator network - G<sub>sample</sub>. The discriminator is trained to maximize the probability of assigning the correct class label to both: real images (from the training set) and fake samples (from G). In the end, the hope is that the game finds an equilibrium - the Nash equilibrium.

In this situation, the Generator would capture the data probability distribution. And the discriminator, in turn, would not be able to distinguish between real or fake samples.

GANs have been used in a lot of different applications in the past few years. Some of them include: generating synthetic data, Image in-paining, semi-supervised learning, super-resolution, text to image generation and more.

However, much of the recent work on GANs is focused on developing techniques to stabilize training. Indeed, GANs are known to be unstable (during training) and very sensitive to the choice of hyper-parameters.

In this context, this piece presents an overview of two relevant techniques for improving GANs. Specifically, we aim to describe recent methods for improving the quality of the Generator's samples. To do that, we address 2 techniques explored in the recent paper: [Self-Attention Generative Adversarial Networks](#1).

All the code developed with the Tensorflow Eager execution API is available [here](https://github.com/sthalles/blog-resources/tree/master/sagan).

## Convolutional GANs

The [Deep Convolutional GAN](#3) (DCGAN)  was a leading step for the success of image generative GANs. DCGANs are a family of ConvNets that impose certain architectural constraints to stabilize the training of GANs. In DCGANs, the generator is composed as a series of transpose convolution operations. These ops, take in a random noise vector z and transforms it by progressively increasing its spatial dimensions while decreasing its feature volume depth.

<figure>
  <img class="img-responsive center-block" src="{{ site.url }}/assets/dcgan/generator-model.png" alt="DCGAN generator network">
</figure>

DCGAN introduced a series of architectural guidelines with the goal of stabilizing the GAN training. To begin, it advocates for the use of strided convolutions instead of pooling layers. Moreover, it uses batch normalization (BN) for both generator and discriminator nets. Finally, it uses ReLU and Tanh activations in the generator and leaky ReLUs in the discriminator.

***Batch norm works by normalizing the input features of a layer to have zero mean and unit variance***. BN was essential for getting Deeper models to work without falling into ***mode collapse***. Mode collapse is the situation in which G creates samples with very low diversity. In other words, G returns the same looking samples for different input signals. Also, batch norm helps to deal with problems due to poor parameters' initialization.

Besides, DCGAN uses Leaky ReLU activations in the discriminator net. ***Different from the regular ReLU function, Leaky ReLU allows the pass of a small gradient signal for negative values***. As a result, it makes the gradients from the discriminator flows stronger into the generator. Instead of passing a gradient (slope) of 0 in the back-prop pass, it passes a small negative gradient.

The architectural guidelines introduced by DCGANs, are still present in the design of recent models. However, much of the work focuses on how to make the GAN training more stable.

## Self-Attention GANs

Self-Attention for Generative Adversarial Networks (SAGANs) is one of these works. Recently, attention techniques have been explored, with success, in problems like Machine Translation. Self-Attention GANs is an architecture that allows the generator to model long-range dependency. The key idea is to make the generator able to produce samples with global detailing information.

If we look at the DCGAN model, we see that regular GANs are heavily based on convolutions. These operations use a local receptive field (the convolutional kernel) to learn representations. ***Convolutions have very nice properties such as parameter sharing and translation invariance***.

A typical Deep ConvNet learns representations in a ***hierarchical fashion***. For a regular image classification ConvNet, simple features like edges and corner are learned in the first few layers. Moreover, ConvNets are able to use these simple representations to learn more complex ones. ***In short, ConvNets learn representations that are expressed in terms of simpler representations***. Consequently, long-range dependency might be hard to learn.

Indeed, it might only be possible for very decimated feature vectors. The problem is that, at this granularity, the amount of signal loss is such that it becomes difficult to model long-range details.

Take a look at these images.

<figure>
  <img class="img-responsive center-block" src="{{ site.url }}/assets/advanced_gans/dcgan_samples.png" alt="DCGAN samples">
</figure>

They are from the DCGAN model trained on ImageNet. As pointed at [Self-Attention GANs](#1), most of the image content that does not exhibit elaborated shapes looks fine. Put it differently, GANs usually do not have problems modeling less structural content like the sky or the ocean.

Nonetheless, the task of creating geometrically complex forms, such as four-legged animals, is far more challenging. That is because, complicated geometrical contours demand long-range details that the convolution, by itself, might not grasp. That is where attention comes into play.

The idea is to give to the generator, information from a broader feature space. Not only the convolutional kernel range. By doing so, the generator can create samples with fine-detail resolution.

### Implementation

In general, given the input features to a convolutional layer L, the first step is to transform L in 3 different representations. We convolve *L* using *1x1* convolutions to get three feature spaces: *f*, *g*, and *h*. Here, we use *f* and *g* to calculate the attention. To do that, we linearly combine *f* and *g* using a matrix multiplication and the result is fed into a softmax layer.

<figure>
  <img class="img-responsive center-block" src="{{ site.url }}/assets/advanced_gans/self_attention_module.png" alt="Self attention module">
</figure>

The resulting tensor is linearly-combined with *h* and finally scaled by *gamma*. Note that *gamma* starts as 0. At the beginning of training, *gamma* cancels out the attention layers. As a consequence, the network only relies on local representations from the regular convolutional layers. However, as *gamma* receives gradient descent updates, the network gradually allows the passage of signals from non-local fields.

Also, note that the feature vectors *f* and *g* have different dimensions than *h*. As a matter of fact, *f* and *g* use 8 times less convolutional filters than *h* does.

That is the complete code for the self attention module.

{% gist 507ce723226274db8097c24c5359d88a %}

## Spectral Normalization

Previously, [Miyato et al](#2) proposed a normalization technique called spectral normalization (SN). In a few words, SN constrains the Lipschitz constant of the convolutional filters. Spectral norm was used as a way to stabilize the training of the discriminator network. In practice, it worked very.

Yet, there is one fundamental problem when training a normalized discriminator. Prior work has shown that regularized discriminators make the GAN training slower. For this reason, some workarounds consist of uneven the rate of update steps between the generator and the discriminator. In other words, we can update the discriminator a few times before updating the generator. For instance, regularized discriminators might require 5 or more update steps for 1 generator update.

To solve the problem of slow learning and imbalanced update steps, there is a simple yet effective approach. It is important to note that in the GAN framework, G and D train together. In this context, [Heusel et al](#5) introduced the two-timescale update rule (TTUR) in the GAN training. It consists of providing different learning rates for optimizing the generator and discriminator.

The discriminator trains with a learning rate 4 times greater than G - 0.004 and 0.001 respectively. A larger learning rate means that the discriminator will absorb a larger part of the gradient signal. Hence, a higher learning rate eases the problem of slow learning of the regularized discriminator. Also, this approach makes it possible to use the same rate of updates for the generator and the discriminator. In fact, we use a 1:1 update interval between generator and discriminator.

Moreover, [[6]](#6) have shown that well-conditioned generators are causally related to GAN performance. Given that, [Self-Attention for GANs](#1) proposed using spectral normalization for stabilizing training of the generator network as well. For G, spectral norm prevents the parameters to get very big and avoids unwanted gradients.

### Implementation

It is important to note that the spectral normalization (SN) algorithm introduced by [Miyato et al](#2) is an ***iterative approximation***. It defines that the spectral norm used to regularize each Conv layer W<sup>l</sup> is the largest singular value of W<sup>l</sup>. Here, *l* belongs to the set *L* of all layers of the network.

Applying singular value decomposition at each step might be computational expansive. Instead, [Miyato et al](#2) uses the ***power iteration method*** to estimate the spectral normal of each layer.

To implement SN using Tensorflow eager execution with the Keras layers, we had to download and tweak the [convolutions.py](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py) file. The complete code can be accessed [here](https://github.com/sthalles/blog-resources/blob/master/sagan/libs/convolutions.py). Bellow we show the juicy parts of the algorithm.

To begin, we randomly initialize a vector *u* as following.

```
self.u = K.random_normal_variable([1, units], 0, 1, dtype=self.dtype)  # [1, out_channels]
```

As shown in Algorithm 1, the power iteration method computes l2-distances between the linear combination of the vector *u* and the convolutional kernels W<sub>i</sub>. Also, the spectral norm is calculated on the unnormalized kernel weights.

<figure>
  <img name="sn_algorithm" class="img-responsive center-block" src="{{ site.url }}/assets/advanced_gans/spectral_norm_algorithm.png" alt="Spectral Norm algorithm">
</figure>

Note that during training, the values of *ũ*, calculated in the power iteration, are used as the initial values of *u* in the next iteration. This  strategy allows the algorithm to get very good estimates using only 1 round of the power iteration. Also, to normalize the kernel weights, we divide them by the current spectral norm estimation.

{% gist f9d770d85011ea49ca2736c3279b0734 %}

## Notes on Implementation

We trained a custom version of the SAGAN model using spectral normalization and self-attention. We used Tensorflow's tf.keras and Eager execution.

The Generator takes a random vector *z* and generates *128x128* RGB images. All layers, including dense layers, use spectral normalization. Additionally, the generator uses batch normalization and ReLU activations. Also, it uses self-attention in between middle-to-high feature maps. Like in the original implementation, we placed the attention layer to act on feature maps with dimensions *32x32*.

The discriminator also uses spectral normalization (all layers). It takes RGB image samples of size *128x128* and outputs an unscaled probability. It uses leaky ReLUs with an alpha parameter of 0.02. Like the generator, it also has a self-attention layer operating of feature maps of dimensions *32x32*.    

<figure>
  <img name="sn_algorithm" class="img-responsive center-block" src="{{ site.url }}/assets/advanced_gans/model_architecture.png" alt="SAGAM model architecture">
</figure>

The goal is to minimize the hinge version of the adversarial loss. To do that, we trained the generator and discriminator in an alternating style using the Adam Optimizer.

<figure>
  <img name="sn_algorithm" class="img-responsive center-block" src="{{ site.url }}/assets/advanced_gans/hinge_gan_loss.png" alt="Spectral Norm algorithm">
</figure>

{% gist 0cf940d807c5d01bf0bb9fa44600924a %}

For this task, we used the [Large-scale CelebFaces Attributes (CelebA)](#7) Dataset.

These are the results.

<figure>
  <img name="sn_algorithm" class="img-responsive center-block" src="{{ site.url }}/assets/advanced_gans/results.png" alt="Spectral Norm algorithm">
</figure>

Thanks for reading!

---

Cite as:

```
@article{
  silva2018advancedgans,
  title={% raw %}{{% endraw %}{{page.title}}{% raw %}}{% endraw %},
  author={% raw %}{{% endraw %}{{site.author}}{% raw %}}{% endraw %},
  journal={% raw %}{{% endraw %}{{site.url}}{% raw %}}{% endraw %},
  year={% raw %}{{% endraw %}{{page.date | date: "%Y"}}{% raw %}}{% endraw %}
  url={% raw %}{{% endraw %}{{site.url}}{{page.url}}{% raw %}}{% endraw %}
}
```