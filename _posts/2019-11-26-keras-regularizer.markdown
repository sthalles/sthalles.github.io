---
layout: post
title:  "How to Add Regularization to Keras Pre-trained Models the Right Way"
date:   2019-11-26 08:00:00 -0300
categories: deep learning
short_description: "If you train deep learning models for a living, you might be tired of knowing one specific and important thing: fine-tuning deep pre-trained models requires a lot of regularization. "
tags: machine-learning deep-learning regularization tensorflow keras
image_url: "/assets/keras-regularizer/cover.jpg"
---

<!-- ## Photo by Kelly Sikkema on Unsplash -->

## Introduction

If you train deep learning models for a living, you might be tired of knowing one specific and important thing:

> **Fine-tuning deep pre-trained models requires a lot of regularization.**

As a contrast, you might have noticed that it is not always obvious how to add regularization to pre-trained models taken from deep learning libraries such as Keras. Also, finding the right answer to this question is not easy either. In the processing of writing this post, I came across many code snippets on Stack Overflow and some blog posts that simply did not get it right. Then, as a way of reducing your Google search (and to help my future self), I am going to show you how to add regularization to pre-trained Keras models in the right way.

Let's start with the basics.

## Fine-Tuning

Fine-tuning is the process of taking a pre-trained model and use it as the starting point to optimizing a different (most of the times related) task. You can imagine taking a ResNet50 model trained on the ImageNet database and use it fit a new problem like insect classification.

The process usually follows simple steps.

1. **We first load the model architecture and pre-trained weights.** For Computer Vision, where transfer-learning is more established, this is where we load one of these famous architectures like DenseNets, or MobileNets and their respective weights (trained on ImageNet).

2. **We then add a task-specific classification layer on the top of the pre-trained model.** This is usually a dense layer with a softmax or sigmoid activation. Note that the number of units in the classification layer has to be equal to the number of classes of your new problem. So, if your insect's dataset contains 28 kinds of bugs and the likes, the last layer needs to have 28 units.

3. Then, we finish up the model preparation. In Keras, we compile the model with an optimizer and a loss function, set up the hyper-parameters, and call fit.

P.S. that might be oversimplified but it is fine for our example.

## Fighting Overfit

One thing we must have in mind is:

>**When fine-tuning pre-trained models, overfitting is a much bigger concern.**

The problem is easy to see. If you have a small training data, you will keep showing the same instances over and over again to the network. Moreover, as we know, pre-trained ConvNets on ImageNet are usually very complex; i.e. they have a lot of training parameters. As a consequence, the model will quickly memorize your training data entirely.

As a solution, fine-tuning usually requires 2 things:
1. **A lot of regularization**
2. **A very small learning rate**

For regularization, anything may help. I usually use l1 or l2 regularization, with early stopping. For ConvNets without batch normalization, [Spatial Dropout](https://www.tensorflow.org/api_docs/python/tf/keras/layers/SpatialDropout2D) is helpful as well

As a side note, deep learning models are known to be data-hungry. These models need a lot of data to disentangle very complex high-dim spaces into linearly separable decisions in the feature space. Many people see fine-tuning as a way of training deep models using smaller datasets. Although in practice this argument may sound right, there is an important catch in here. Even though you may be able to fit a new model using a much smaller dataset, remember that your pre-trained model trained for DAYS (using multiple GPUs). Put it differently, fine-tuning actually means -- ***Standing on the Shoulder of Giants.***

Let's now jump into the code.

## Hacking Keras

Intuitively, the process of adding regularization is straightforward. After loading our pre-trained model, refer to as the ***base model***, we are going loop over all of its layers. For each layer, we check if it supports regularization, and if it does, we add it. The code looks like this.

{% gist bff8628356cbc5ff242e711251f5f8bd %}


It looks like we are done. Indeed, if you Google how to add regularization to Keras pre-trained models, you will find the same.

As a safety check, let's make sure that regularization is properly set. In Keras, we can retrieve losses by accessing the losses property of a Layer or a Model. In our case, we can access the list of all losses (from all Layers with regularization) by:

{% gist c186f3e98aa6913d7bfe2fafbe418976 %}

*P.S. if you're confused with the nomenclature, the property is called losses, because the regularization penalties are added to the loss function during optimization.*

As you can see, there is something weird going on. **The list is just empty, which means, there no regularization penalty applied to the convolutional kernels.**

But what just happened?

Well, going straight to the problem, it seems that when we change a property on a layer, as we did, **the only thing that actually changed was the model config.** Thus, the model object itself is just as it was when we loaded. No changes at all.

Take a look at the config file after adding regularization. The ***kernel_regularizer*** property is there like we set it.

{% gist c18f574ecbc0c1cb2b6237d9503279a6 %}

One simple solution to this problem is to reload the model config. This is easy to do and solves the problem.

{% gist 605ff1280207b6f798ca5d9946c6f710 %}

Now, if we attempt to see the ***model.losses***, there we have it.

{% gist 9fa0dc384d38409341c323896f19b5ff %}

However, as a common hacking, this introduces another problem. If you pay closer attention to the model's weights, after reloading the model from the config file, the weights got reset! ***We just lost, all of the ImageNet pre-trained parameters!***

Well, a quick solution is to use the same strategy. We can save the weights, before reloading the model config, and reload the weights after the model is properly loaded.

The function below does the complete job. You can pass any model from [Keras Applications](https://keras.io/applications/) ***(using Tensorflow 2.0)***, along with the regularizer you want, and it returns the model properly configured. Note how we save and reload the model weights before and after reloading the model from the config file.

Also, you can add ***bias_regularizer*** and ***activity_regularizer*** using the same code.

{% gist d4e0c4691dc2be2497ba1cdbfe3bc2eb %}

And that is it. A quick but hopefully, useful trick to regularize your pre-trained models.

**Thanks for reading!**

---

Cite as:

```
@article{
  silva2019kerasregularization,
  title={% raw %}{{% endraw %}{{page.title}}{% raw %}}{% endraw %},
  author={% raw %}{{% endraw %}{{site.author}}{% raw %}}{% endraw %},
  journal={% raw %}{{% endraw %}{{site.url}}{% raw %}}{% endraw %},
  year={% raw %}{{% endraw %}{{page.date | date: "%Y"}}{% raw %}}{% endraw %}
  url={% raw %}{{% endraw %}{{site.url}}{{page.url}}{% raw %}}{% endraw %}
}
```