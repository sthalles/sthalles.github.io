---
layout: post
title:  "Deep Semantic Image Segmentation Network"
date:   2019-01-01 08:00:00 -0300
categories: deep learning
short_description: "DenseNets offer very scalable models that achieve very good accuracy and are easy to train. The key idea consists sharing feature maps within a block through direct connections between layers."
image_url: "/assets/deep_segmentation_network/semantic_segmentation.jpg"
---
## Introduction

Deep Convolution Neural Networks (DCNNs) have achieved remarkable success in various Computer Vision applications. Like others, the task of semantic segmentation is not an exception to this trend.

This piece provides an introduction to Semantic Segmentation with a hands-on TensorFlow implementation. We go over one of the most relevant papers on Semantic Segmentation of general objects - [Deeplab_v3](https://arxiv.org/abs/1706.05587). You can clone the notebook for this post [here](https://github.com/sthalles/deeplab_v3).

## Semantic Segmentation

Regular image classification DCNNs have similar structure. These models take images as input and output a single value representing a category label. To do that, the input images pass through a Deep Convolution Neural Network (DCNN).

Usually, image classification DCNNs have three main components. Convolutions, pooling, and fully connected layers. In the end, the network outputs a feature vector containing the probabilities for each class label. Note that, here, we categorise an image as whole. In other words, we assign a single label to an entire image.

<figure>
  <img class="img-responsive center-block" src="{{ site.url }}/assets/deep_segmentation_network/cnns.jpg" alt="ResNet bottleneck layer">
  <figcaption class="caption center">VGG-16 architecture overview. </figcaption>
</figure>

Different from image classification, in semantic segmentation we want to make decisions for every pixel in an image. So, for each pixel, the model needs to classify it as one of the pre-determined classes. Put another way, semantic segmentation means understanding images in a pixel level.

Semantic segmentation doesn't differentiate between instances. Because we try to classify each pixel as an individual label, if we have two objects of the same class, they end up having the same category label. Instance Segmentation is the class of problems where we need to differentiate between instances.

To use DCNNs for semantic segmentation, one very important concept is the fully convolution network.

As an example, instead of having pooling and fully connected layers, imagine passing an image through a series of convolutions. We can set each convolution to have stride 1 and "SAME" padding. Doing this, we get the benefit of having the output vector with the same dimensions as the input. In other words, each convolution preserves the spatial dimensions of its input. In summary, we could stack a bunch of these convolutions and have our segmentation model.

<figure>
  <img class="img-responsive center-block" src="{{ site.url }}/assets/deep_segmentation_network/fully_conv.png" alt="ResNet bottleneck layer">
  <figcaption class="caption center">... </figcaption>
</figure>

This model could output a probability tensor with shape [W,H,C], where W and H represents the Width and Height. And C the number of class labels. With this tensor, we can apply an argmax function (on the second axis) and get a tensor shape like [W,H,1]. After that, we can compute the cross entropy loss between each pixel of the true labels and our predictions. Finally, we can average that value and train the network using back prop to minimize this error.

There is one problem with this approach though. As we mentioned, using convolutions with stride 1 and "SAME" padding preserves the input dimensions. However, preserving dimensions like that would make the model super expensive in both ways. Memory consumption and computation needs.

To ease that problem, segmentation networks usually have three main components. Convolutions, downsampling, and upsampling layers.

<figure>
  <img class="img-responsive center-block" src="{{ site.url }}/assets/deep_segmentation_network/encoder-decoder.png" alt="ResNet bottleneck layer">
  <figcaption class="caption center">... </figcaption>
</figure>

Basically, there are two common ways to do downsampling in neural nets. By convolution striding or from regular pooling operations. In general, downsampling has one goal. To reduce the spatial dimensions of given feature maps. Yet, they do it in detriment of losing some features in the process. For that reason, downsampling allows us to perform deeper convolutions without much memory concerns.

Also, note that the first part of this architecture looks a lot like the usual classification DCNNs. With one exception, they do not put in place fully connected layers. Up to this point, we have a feature vector with shape [w,h,d]. w and h are the width and height of the tensor, and d is its depth. Note that the spatial dimensions of this output vector is smaller than the original input.

<figure>
  <img class="img-responsive center-block" src="{{ site.url }}/assets/deep_segmentation_network/fully_conv_network.png" alt="ResNet bottleneck layer">
  <figcaption class="caption center">(Top) VGG-16 network on its original form. Note the 3 fully-connected layers on top of the convolution stack. (Down) VGG-16 model when substituting its fully-connected layers to 1x1 convolutions. This change allows the network to output a coarse heat-map. </figcaption>
</figure>

Now, instead of outputting a dense feature vector that explains the whole image, we feed the output to a series of upsampling layers. These layers work on reconstructing the output of the first part of the network. The goal is to increase the spatial resolution so our output image has the same dimensions as the input.

Usually, upsampling layers are based on strided transpose convolutions. These functions go from deep and narrow layers to wider and shallower ones. In short, they are used to increase feature maps to a desired spatial dimension.

In most papers, the two components of a segmentation network are called: encoder and decoder. In short, the first, "encodes" its information into a compressed vector used to represent its input. The second (the decoder) works on reconstructing this signal to the desired outcome.

There are many network implementations based on encoder-decoder architectures. FCNs, SegNet and UNet are some of the most popular ones. As a result, we have seen many successful segmentation models in a variety of fields.

## Architecture

Different from most encoder-decoder designs, Deeplab offers a different approach to semantic segmentation. It presents an architecture for controlling signal decimation and learning multi-scale contextual features.

<figure>
  <img class="img-responsive center-block" src="{{ site.url }}/assets/deep_segmentation_network/network_architecture.png" alt="ResNet bottleneck layer">
</figure>

The Deeplab segmentation network uses the ResNet as its main design module. However, it proposes a new Residual block for multi-scale feature learning. In fact, the authors took the last ResNet block, denoted as block4, and made some modifications to it. Instead of regular convolutions and max pooling layers, the new block uses atrous convolutions. Within this block, it uses different dilation rates to capture multi-scale context.

Also, on top of this new model, it uses Atrous Spatial Pyramid Pooling (ASPP). ASPP uses dilated convolutions with different rates as an attempt of classifying regions of an arbitrary scale. Let’s go over each one of the main components.

To understand the deeplab architecture, we need to focus on three components. (i) The ResNet architecture, (ii) atrous (dilated) convolutions and (iii) Atrous Spatial Pyramid Pooling (ASPP). Let’s go over each one of them.

### ResNets

ResNet is a very popular DCNN that won the ILSVRC 2015 classification task. One of the main contributions of ResNets was to provide a framework to ease the training of deeper models.

In its original form, ResNets contain 4 computational blocks. Each block contains a different number of **Residual Units**. These units perform a series of convolutions in a special way. Also, each block is intercalated with max-pooling operations to reduce spatial dimensions.

The original paper presents two types of Residual Units. The *baseline* and the *bottleneck* blocks.

The baseline unit contains two *3x3* convolutions with Batch Normalization(BN) and ReLU activations.

<figure>
  <img class="img-responsive center-block" src="{{ site.url }}/assets/deep_segmentation_network/resnet_residual_units.png" alt="ResNet bottleneck layer">
  <figcaption class="caption center">Figure 3: ResNet building blocks. (left) baseline; (right) bottleneck unit</figcaption>
</figure>

The second, the bottleneck unit, consists of three stacked operations. A series of *1×1*, *3x3* and *1x1* convolutions substitute the previous design. The two *1x1* operations are designed for reducing and restoring dimensions. This leaves the 3x3 convolution, in the middle, to operate on a less dense feature vector. Also, BN is applied after each convolution and before ReLU non-linearity.

To help understanding, let's denote these group of operations as a function *F* of its input *x*.

After the non-linear transformations in *F(x)*, its result is combined with the original input x. This combination is done by adding the two functions. Connecting the original input *x* with the non-linear function *F(x)* shows some advantages. It allows the gradient signal from later layers to be sent to earlier layers. Also, skipping the operations on *F(x)* allows the later layers to have access to a stronger gradient signal. As a result, this type of connectivity has been shown to ease the training of deeper networks.

Non-bottleneck units also show gain in accuracy as we increase model capacity. Yet, *bottleneck* residual units stands some practical advantages. First, it performs more computations carrying almost the same number of parameters. Second, they also perform in a similar computational complexity as its counterpart.

In practice, *bottleneck* units are more suitable for training deeper models because of less training time and computational resources need.

For our implementation though, we use the full pre-activatiom Residual Unit. The only difference, from the standard bottleneck layer, lies on the order in which BN and ReLU activations are placed. For the full pre-activatiom, BN and ReLU (in this order) occur before convolutions.

<figure>
  <img class="img-responsive center-block" src="{{ site.url }}/assets/deep_segmentation_network/various_resnet_based_blocks.png" alt="ResNet bottleneck layer">
  <figcaption class="caption center">Figure 4: ResNet building block architectures. (Left-most) the original ResNet block. (Right-most) the improved full pre-activation version. </figcaption>
</figure>

As shown in Identity Mappings in Deep Residual Networks, the full pre-activation unit performs better than other variants.
*Note that the only difference among these designs is the order in which BN and RELu are performed.*

### Atrous Convolutions

Atrous (or dilated) convolutions are regular convolutions with a factor that allows us to expand the filter's field of view.

Consider a *3x3* convolution filter for instance. When the dilation rate is equal to 1, it behaves like a standard convolution. If we set the rate factor to 2, it has the effect of enlarging the convolution kernel.

In theory, it works like that. First, it expands (dilates) the convolution filter according to the dilation rate. Second, it fills the empty spaces with zeros - creating an sparse like filter. Finally, it performs regular convolution with this dilated filter.

<figure>
  <img class="img-responsive center-block" src="{{ site.url }}/assets/deep_segmentation_network/atrous_conv.png" alt="ResNet bottleneck layer">
  <figcaption class="caption center">Figure 5: Atrous convolutions with various rates. </figcaption>
</figure>

As a result, applying a dilation rate = 2 to a regular 3x3 convolution filter, has the effect of **expanding** the filter to cover an area equivalent to a 5x5. Yet, because it acts like a sparse filter, only the original *3x3* cells will do computation and produce results. I said "act" because most frameworks don't implement atrous convolutions using sparse filters - because of memory concerns.

In a similar way, setting the atrous rate to 3 allows a regular *3x3* convolution to get signals from a larger area. In fact, it expands to a *7x7* corresponding area.

This effect allow us to control the resolution at which we compute feature responses. Also, atrous convolution adds larger context without increasing the number of parameters or the amount of computations.

Deeplab also shows that the dilation rate must be tuned according to the size of the feature maps. They studied the effects of using large dilation rates over small feature maps.

<figure>
  <img class="img-responsive center-block" src="{{ site.url }}/assets/deep_segmentation_network/extreme_dilation.png" alt="ResNet bottleneck layer">
  <figcaption class="caption center">Figure 6: For a 14x14 input image, a 3x3 filter with dilation rate of 14 makes the atrous convolution behaves like a regular 1x1 convolution.</figcaption>
</figure>

When the dilation rate is very close to the feature map's size, a regular *3x3* atrous filter acts as a standard *1x1* convolution.

To avoid this problem, the output stride of the ResNet model is set to 16. In other words, for input images of size 513x513, the block that uses atrous convolutions operates on feature maps of size 32x32.

Also, because the atrous block does't implement downsampling, ASPP also runs on the same feature response size. As a result, it allows learning features from multi-scale context using relative large dilation rates.

Each unit has a 3x3 convolution resulting in a total of three 3x3 convolutions.

The last ResNet block contains three residual units. In total, the 3 units have three 3x3 convolutions. Motivated by *multigrid* methods, Deeplab proposes different dilation rates for each convolution. In summary, *multigrid* defines the dilation rates for the three convolutions.

In practice:

For block4, when output stride = 16 and Multi Grid = (1, 2, 4), the three convolutions will have rates = 2 · (1, 2, 4) = (2, 4, 8) respectively.

### Atrous Spatial Pyramid Pooling

For ASPP, the idea is to provide the model with multi-scale information. To do that, ASPP adds a series atrous convolutions with different dilation rates. These rates are designed to capture long range context. Also, to add global context information, ASPP incorporates image-level features via Global Average Pooling (GAP).

In the version, ASPP contains 4 parallel operations. These are a *1x1* convolution and three *3x3* convolutions with *dilation rates =(6,12,18)*. As we mentioned, at this point, the feature maps' nominal stride is equal to 16.

As described earlier, the efficiency of atrous convolutions depends on a good choice of the dilation rates. Because of that, it is important to know the concept of **output stride** in neural networks.
***Output stride explains the ratio of the input image size to the output feature map size. It defines how much signal decimation the input vector suffers as it passes the network.***

For an output stride of 16, an image size of 224x224x3 outputs a feature vector with 16 times smaller dimensions. That is 14x14.

Based on the original implementation, we used image sizes of 513x513 during training. Thus, using an output stride 16 means that ASPP receives feature vectors with size 32x32.

The paper also describes a technique for improving older versions of ASPP. The idea is to add global context information using image-level features.

First, it applies GAP to the output features from the last atrous block. Second, the resulting features are fed to a *1x1* convolution with *256 filters*. Finally, the result is bilinearly upsampled to the correct dimensions.

In the end, the features, from all the branches, are combined to a single vector via concatenation. The output is then convolved with another 1x1 kernel - using BN and 256 filters.

{% highlight python %}
@slim.add_arg_scope
def atrous_spatial_pyramid_pooling(net, scope, depth=256):
    """
    ASPP consists of (a) one 1×1 convolution and three 3×3 convolutions with rates = (6, 12, 18) when output stride = 16
    (all with 256 filters and batch normalization), and (b) the image-level features as described in https://arxiv.org/abs/1706.05587
    :param net: tensor of shape [BATCH_SIZE, WIDTH, HEIGHT, DEPTH]
    :param scope: scope name of the aspp layer
    :return: network layer with aspp applyed to it.
    """

    with tf.variable_scope(scope):
        feature_map_size = tf.shape(net)

        # apply global average pooling
        image_level_features = tf.reduce_mean(net, [1, 2], name='image_level_global_pool', keep_dims=True)
        image_level_features = slim.conv2d(image_level_features, depth, [1, 1], scope="image_level_conv_1x1", activation_fn=None)
        image_level_features = tf.image.resize_bilinear(image_level_features, (feature_map_size[1], feature_map_size[2]))

        at_pool1x1 = slim.conv2d(net, depth, [1, 1], scope="conv_1x1_0", activation_fn=None)

        at_pool3x3_1 = slim.conv2d(net, depth, [3, 3], scope="conv_3x3_1", rate=6, activation_fn=None)

        at_pool3x3_2 = slim.conv2d(net, depth, [3, 3], scope="conv_3x3_2", rate=12, activation_fn=None)

        at_pool3x3_3 = slim.conv2d(net, depth, [3, 3], scope="conv_3x3_3", rate=18, activation_fn=None)

        net = tf.concat((image_level_features, at_pool1x1, at_pool3x3_1, at_pool3x3_2, at_pool3x3_3), axis=3,
                        name="concat")
        net = slim.conv2d(net, depth, [1, 1], scope="conv_1x1_output", activation_fn=None)
        return net
{% endhighlight %}

After ASPP, we feed the result to another 1x1 convolution - to produce the final segmentation logits.

Besides, Deeplab also debates the effects of different output strides on segmentation models. It argues that excessive signal decimation is harmful for dense prediction tasks. In summary, models with smaller output stride - less signal decimation - tends to output finer segmentation maps. Yet, training models with smaller output stride demands more training time.

## Implementation Details

To create deeplab_v3, we used the Tensorflow Slim package along with the ResNet-50 implementation available at [Tensorflow-Slim Github](https://github.com/tensorflow/models/tree/master/research/slim/nets). Following the best results reported in the [Deeplab_v3](https://arxiv.org/pdf/1704.06857), this implementation employs the following network configuration:

-          *output stride = 16*
-          *Fixed multi-grid atrous convolution rates as (1,2,4) to the last ResNet block (block 4).*
-          *Employ ASPP with rates (6,12,18) after the last atrous convolution block.*

Setting the *output stride* to 16 gives us the advantage of substantially faster training compared to output stride = 8 because for for the smaller output stride, the last block deals with 4 times larger feature maps than its counterpart.

The multi-grid dilation rates are applied to the 3 Residual Units that compose the last ResNet block. Finally, the ASPP rates are applied to the three parallel *3x3* atrous convolution branches that compose the ASPP layer added before the last *1x1* convolution that generates the final logits.

The final logits shape has a 16 times smaller dimension compared to the input image shape. We trained the network on the PASCAL VOC 2012 segmentation dataset. Based on the original implementation, we feed the network with batches of *513x513* which, after a forward pass, gives a segmentation map of size *33x33*. Before computing the *cross-entropy* error, we upsample the logits to its original size so that it can be compared (pixel-wise) to the image label. After, the gradient error will be propagated back to update the network parameters.

## Concluding

DenseNets offer very scalable models that achieve very good accuracy and are easy to train. The key idea consists sharing feature maps within a block through direct connections between layers. Moreover, it demands fewer parameters than a number of other models like ResNets, Inception Networks, and others while offering equally or improved accuracy on various classification datasets. The DenseNet library described here implements all 4 architectures used to train on ImageNet plus a custom constructor in which any network variation can be experimented. Feel free to checkout the [code on Github](https://github.com/sthalles/dense-net) and make pull requests with any suggestion.
