---
layout: post
title:  "Densely Connected Convolutional Networks in Tensorflow"
date:   2017-12-10 08:00:00 -0300
categories: deep learning
short_description: "Supervised learning has been the center of most researching in deep learning in recent years. However, the necessity of creating models capable of learning from fewer or no labeled data is greater year by year."
image_url: "/assets/semi-supervised/GAN_classifier.png"
---

## Intro

If you like Neural Nets, you certainly have heard about VGGNet, Resnet, Inception, and others. These networks, each one in its time, reached state-of-the-art performance in some of the most famous challanges in Computer Vision. If we look at the short and successfull history of Deep Neural Networks, post the GPGPU and big data revolution, we notice that year by year, these models got more deeper and more powerful. However, as networks were getting more and more dense in the number of paramenters and layers, the problem of how to prevent the gradient from vanishing by the time it reaches the first layers of the network was something to worry about.

To address this issue, many network architectures emerged such as Resnet and Highway networks. Besides some changes, all of them tried to solve this problem using one single approach - ***create shortcut connections that bypasses a group of operations so that the gradient signal could be propagated without much loss from the end to the begining of the netowork***.

In this context, arouse the Densely Connected Convolutional Networks, DenseNets.

I have been using this architecute for a while in at least two different kinds of problems, classification and densely prediction problems such as semantic segmentation. During this time, I developed a library to use DenseNets using Tensorflow with its Slim package. I really like this model and in this post, we are going to do an overview of this archtecture, compare it with other very popular ones, and show how one might use the Library for its own pleasure.

## Archtecture

To understand DenseNets, we need to focus on two principal components of its archtecure. The Dense Block, and the Transition Layer. A DenseNet is basically a stack of dense blocks followed by a transition layers. Each block consists of a series of units, each unit packs two convolution operations, each of which is preceeded by Batch Normalization and ReLU activations. In addition, each unit outputs only k feature vectors. This parameter k, also described as the growth rate, controls how much new information the layers allow to pass through.

On the other hand, transition layers are very simple components designed to perform downsampling of the features passing the network. Every transition layer consisits of a Batch Normalization layer, followed by a 1x1 convolution, followed by a 2x2 average pooling.

<figure>
  <img class="img-responsive center-block" src="{{ site.url }}/assets/densenets/densetnet_block_unit_and_transitions_layer.png" alt="DenseNet block unit and transition layer">
  <figcaption class="caption center"> (Left) DenseNet Block unit operations. (Right) DenseNet Transitions Layer </figcaption>
</figure>

The big difference however, from other regular CNNs, is that each unit within a dense block is connected to every other unit bofore it. In summary, within a given block, the nth unit, receices as input the feature-maps learned by the n-1th, n-2th all the way down to the 1st unit in the pipeline. As you might guess, ***it allows DenseNet models to carry very few parameters because there is a very high level of feature sharring amongst the units.*** We will talk more about number of parameters in a bit.

Different then the ResNet archtecture, DenseNets propose feature reuse among units by concatenation. As a consequence of that archtecure choice, DenseNet models tend to be more compact (in the number of parameters) than ResNets because every feature-map learned by any given DenseNet unit is reused by all of the following units within a block, avoinding the relearning of these same features. To get a better glance at it, let's have a look at the differences between a Resnet unit and a DenseNet unit. Both archtectures employ the so called bottleneck layer, where there is a 1x1 convolution designed to reduce the spatial dimentionallity, followed by a more wider convolution, in this case a 3x3 operation for feature learning.

In its original form, the ResNet bottleneck layer consists of a 1x1 followed by a 3x3 followed by another 1x1 convolution, closing with an addition operation between the original input and the result of the non-linear transformations. This very elegant design gave the ResNet the ILSVRC 2015 classification task challenge chanpionship and since then, it inspired many others similar archtectures that improved upon it.

## Generator

The Generator follows a very standard implementation described in the DCGAN paper. This approach consists of reshaping a random vector z to have a 4D shape and then feed it to a sequence of transpose convolutions, batch normalization and leaky RELU operations that increase the spatial dimensions of the input vector while decreases the number of channels. As a result, the network outputs a 32x32x3 RGB tensor shape that is squashed between values of -1 and 1 through the Hyperbolic Tangent Function.

{% highlight python %}
def generator(z, output_dim, reuse=False, alpha=0.2, training=True, size_mult=128):
    with tf.variable_scope('generator', reuse=reuse):
        # First fully connected layer
        x1 = tf.layers.dense(z, 4 * 4 * size_mult * 4)
        # Reshape it to start the convolutional stack
        x1 = tf.reshape(x1, (-1, 4, 4, size_mult * 4))
        x1 = tf.layers.batch_normalization(x1, training=training)
        x1 = tf.maximum(alpha * x1, x1)

        x2 = tf.layers.conv2d_transpose(x1, size_mult * 2, 5, strides=2, padding='same')
        x2 = tf.layers.batch_normalization(x2, training=training)
        x2 = tf.maximum(alpha * x2, x2)

        x3 = tf.layers.conv2d_transpose(x2, size_mult, 5, strides=2, padding='same')
        x3 = tf.layers.batch_normalization(x3, training=training)
        x3 = tf.maximum(alpha * x3, x3)

        # Output layer
        logits = tf.layers.conv2d_transpose(x3, output_dim, 5, strides=2, padding='same')

        out = tf.tanh(logits)

        return out
{% endhighlight %}

## Discriminator
The discriminator, now a multi class classifier, is the most relevant network for this architecture. Here we setup an also similar DCGAN architecture in which we use a stack of strided 2 convolutions for dimensionality reduction and batch normalization for stabilizing learning (except for the first layer of the network).

The 2D convolution window (kernel or filter) is set to have a width and height of 3 across all the convolution operations. Also, note that we have some layers with dropout. It is important to understand that our discriminator behaves (in part) like any other regular classifier and because of that; it may suffer from the same problems any classifier would if not well designed.

One of the most likely drawback one might encounter when training a big classifier on a very limited labeled dataset is the imminence of overfitting. One thing to watch on “over trained” classifiers is that they typically show a notably difference between the training error (smaller) and the testing error (higher). ***This situation shows that the model did a good job capturing the structure of the training dataset but at the same time, because the model believed too much in the training data, it fails to generalize what it has learned for unseen examples. To prevent that, we make an extensive usage of dropout even for the first layer of the network.***

{% highlight python %}
def discriminator(x, reuse=False, alpha=0.2, drop_rate=0., num_classes=10, size_mult=64):
    with tf.variable_scope('discriminator', reuse=reuse):
        x = tf.layers.dropout(x, rate=drop_rate/2.5)

        # Input layer is ?x32x32x3
        x1 = tf.layers.conv2d(x, size_mult, 3, strides=2, padding='same')
        relu1 = tf.maximum(alpha * x1, x1)
        relu1 = tf.layers.dropout(relu1, rate=drop_rate) # [?x16x16x?]

        x2 = tf.layers.conv2d(relu1, size_mult, 3, strides=2, padding='same')
        bn2 = tf.layers.batch_normalization(x2, training=True) # [?x8x8x?]
        relu2 = tf.maximum(alpha * bn2, bn2)

        x3 = tf.layers.conv2d(relu2, size_mult, 3, strides=2, padding='same') # [?x4x4x?]
        bn3 = tf.layers.batch_normalization(x3, training=True)
        relu3 = tf.maximum(alpha * bn3, bn3)
        relu3 = tf.layers.dropout(relu3, rate=drop_rate)

        x4 = tf.layers.conv2d(relu3, 2 * size_mult, 3, strides=1, padding='same') # [?x4x4x?]
        bn4 = tf.layers.batch_normalization(x4, training=True)
        relu4 = tf.maximum(alpha * bn4, bn4)

        x5 = tf.layers.conv2d(relu4, 2 * size_mult, 3, strides=1, padding='same') # [?x4x4x?]
        bn5 = tf.layers.batch_normalization(x5, training=True)
        relu5 = tf.maximum(alpha * bn5, bn5)

        x6 = tf.layers.conv2d(relu5, 2 * size_mult, 3, strides=2, padding='same') # [?x2x2x?]
        bn6 = tf.layers.batch_normalization(x6, training=True)
        relu6 = tf.maximum(alpha * bn6, bn6)
        relu6 = tf.layers.dropout(relu6, rate=drop_rate)
...
{% endhighlight %}

After a series of convolutions, batch normalization, leaky RELUs and dropout, instead of directly applying a fully connected layer on top of the convolutions, we perform a Global Average Pooling (GAP) operation. Global Average Pooling is a regularization technique that has been used with success in some convolutional classifier nets as a replacement to fully connected layers. In GAP we take the average over the spatial dimensions of a feature map resulting in one value.

{% highlight python %}
...
# Flatten it by global average pooling
# In global average pooling, for every feature map we take the average over all the spatial
# domain and return a single value
# In: [BATCH_SIZE,HEIGHT X WIDTH X CHANNELS] --> [BATCH_SIZE, CHANNELS]
features = tf.reduce_mean(relu7, axis=[1,2])

# Set class_logits to be the inputs to a softmax distribution over the different classes
class_logits = tf.layers.dense(features, num_classes)
...
{% endhighlight %}

For instance, suppose that after a series of convolutions, we get a tensor of shape *[BATCH_SIZE, 8, 8, CHANNELS]* and we want to apply GAP to flat this feature map. GAP works by taking the average value over the *[8x8]* tensor slice resulting in a tensor of shape *[BATCH_SIZE, 1, 1, CHANNELS]* that can be reshaped to *[BATCH_SIZE, CHANNELS]*.

In [Network in network](https://arxiv.org/abs/1312.4400), the authors describe several advantages over traditional fully connected layers such as a higher robustness for spatial translation of the input and less overfitting concerns. After the GAP operation, a fully connected layer is applied to output a feature map that corresponds to the number of classes we want to predict.

Once we get logits associated with the different classes, putting these logits through a softmax gives us the classification probabilities over the classes we want to predict. However, we still need a way to represent the probability of an input image being real rather than fake, that is, we still need to model the binary classification problem for a regular GAN model.

{% highlight python %}
...
# Get the probability that the input is real rather than fake
out = tf.nn.softmax(class_logits) # class probabilities for the 10 real classes
...
{% endhighlight %}

We can think that the logits are in terms of softmax logits and we need to represent them as sigmoid logits as well. Given that the probability of an input being real corresponds to the sum over the real class logits, we can feed these values into a **LogSumExp** function that will model the binary classification value to be fed to a sigmoid function. In order to avoid numerical stabilization problems with the LogSumExp function, we can use the Tensorflow built-in function that will prevent numerical issues that may occur when LogSumExp encounters very extreme either positive or negative values.

{% highlight python %}
...
# This function is more numerically stable than log(sum(exp(input))).
# It avoids overflows caused by taking the exp of large inputs and underflows
# caused by taking the log of small inputs.
gan_logits = tf.reduce_logsumexp(class_logits, 1)
...
{% endhighlight %}

## Model Loss
We can divide the discriminator loss into two parts, one that represents the GAN problem, or the ***unsupervised loss***, and the other that computes the individual real class probabilities, the ***supervised loss***.

For the unsupervised loss, as we mentioned, the discriminator still has to differentiate between real training images and fake images from the generator. As for a regular GAN, half of the time the discriminator receives unlabeled images from the training set and the other half, imaginary unlabeled images from the generator. Since in both cases we are dealing with a binary classification problem in which we want a probability value near 1 for real images and near 0 for unreal images, we use the ***sigmoid cross entropy*** function to compute the loss.

For images coming from the training set, we maximize their probabilities of being real by assigning labels of 1s, and for fabricated images coming from the generator, we maximize their probabilities to be fake by giving them labels of 0s.

{% highlight python %}
...
# Here we compute `d_loss`, the loss for the discriminator.
# This should combine two different losses:
# 1. The loss for the GAN problem, where we minimize the cross-entropy for the binary
#    real-vs-fake classification problem.
tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=gan_logits_on_data,
                                                        labels=tf.ones_like(gan_logits_on_data) * (1 - smooth)))

fake_data_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=gan_logits_on_samples,
                                                     labels=tf.zeros_like(gan_logits_on_samples)))

# This way, the unsupervised
unsupervised_loss = real_data_loss + fake_data_loss
...
{% endhighlight %}

For the supervised loss, we need to use the logits from the discriminator, and since this is a multiclass classification problem, we perform a ***softmax cross entropy*** using the real labels we have available. Note that this part is similar to any other classification model. In the end, the discriminator loss is the sum of both the supervised loss and the unsupervised loss. Also, note that because we are pretending we do not have most of the labels, we need to ignore them in the supervised loss. To do that, we just multiply the loss by the masks variable that basically indicates which set of labels are available for usage.

{% highlight python %}
#  2. The loss for the SVHN digit classification problem, where we minimize the cross-entropy
#     for the multi-class softmax. For this one we use the labels. Don't forget to ignore
#     use `label_mask` to ignore the examples that we are pretending are unlabeled for the
#     semi-supervised learning problem.
y = tf.squeeze(y)
suppervised_loss = tf.nn.softmax_cross_entropy_with_logits(logits=class_logits_on_data,
                                                              labels=tf.one_hot(y, num_classes, dtype=tf.float32))

label_mask = tf.squeeze(tf.to_float(label_mask))

# ignore the labels that we pretend does not exist for the loss
suppervised_loss = tf.reduce_sum(tf.multiply(suppervised_loss, label_mask))

# get the mean
suppervised_loss = suppervised_loss / tf.maximum(1.0, tf.reduce_sum(label_mask))
d_loss = unsupervised_loss + suppervised_loss
{% endhighlight %}


For the generator loss, as described in the [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498) paper, we use feature matching. As the authors describe, feature matching is the concept of “penalizing the mean absolute error between the average value of some set of features on the training data and the average values of that set of features on the generated samples”.

To do that, we take some set of statistics (the moments) from two different sources and force them to be similar. First, we take the average values of the features extracted from the discriminator when a real training minibatch is being processed. Second, we compute the moments in the same way but now for when a minibatch composed of fake images that come from the generator was being analyzed by the discriminator, and finally, with these two sets of moments, the generator loss is the mean absolute difference between them. In other words, as the paper emphasizes, “we train the generator to match the expected values of the features on an intermediate layer of the discriminator”.

{% highlight python %}
# Here we set `g_loss` to the "feature matching" loss invented by Tim Salimans at OpenAI.
# This loss consists of minimizing the absolute difference between the expected features
# on the data and the expected features on the generated samples.
# This loss works better for semi-supervised learning than the tradition GAN losses.

# Make the Generator output features that are on average similar to the features
# that are found by applying the real data to the discriminator

data_moments = tf.reduce_mean(data_features, axis=0)
sample_moments = tf.reduce_mean(sample_features, axis=0)
g_loss = tf.reduce_mean(tf.abs(data_moments - sample_moments))

pred_class = tf.cast(tf.argmax(class_logits_on_data, 1), tf.int32)
eq = tf.equal(tf.squeeze(y), pred_class)
correct = tf.reduce_sum(tf.to_float(eq))
masked_correct = tf.reduce_sum(label_mask * tf.to_float(eq))
{% endhighlight %}

An interesting curiosity about feature matching is that, although it performs well on the task of semi-supervised learning, you will notice that the images generated by the generator network are not as good as the ones created in the last post.

<figure>
  <img class="img-responsive center-block" src="{{ site.url }}/assets/semi-supervised/generated_images.PNG" alt="Fake ticket">
  <figcaption class="caption center"> Sample images created by the generator network using the feature matching loss. </figcaption>
</figure>

In the [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498) paper, OpenAI reports state-of-the-art results for semi-supervised classification learning on MNIST, CIFAR-10 and SVHN. This implementation reaches ***train error of nearly 93%*** and a ***test error of roughly 67 to 68%***. Although the results might seem good, they are actually a little better than the ones reported in this [NIPS 2014](https://arxiv.org/abs/1406.5298) paper which got something around 64%. Note that this notebook is not intended for demonstrating best practices such cross-validation techniques and it only uses some of the techniques described in the paper. The notebook was implemented from the one provided in the Udacity Deep learning Fundamentals nanodegree program in which I am graduated from.

## Concluding

As machine learning grows, attempts to solve already establish problems using less labeled data are key for breaking the unsupervised learning obstacles machine learning is facing right now. In this scenario, GANs pose a real alternative for learning complicated tasks with less labeled samples. However, the distance between semi-supervised and fully supervised learning solutions is still far from being equal, but we certainly can expect this gap to become shorter as new approaches come in to play.  
