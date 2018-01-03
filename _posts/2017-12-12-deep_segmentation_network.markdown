---
layout: post
title:  "Deep Semantic Image Segmentation Network"
date:   2018-01-30 08:00:00 -0300
categories: deep learning
short_description: "DenseNets offer very scalable models that achieve very good accuracy and are easy to train. The key idea consists sharing feature maps within a block through direct connections between layers."
image_url: "/assets/deep_segmentation_network/semantic_segmentation.jpg"
---
## Introduction

Since Deep Convolutional Neural Networks (DCNNs) made their way back and started breaking every record in most of the Computer Vision (CV) applications, the task of Semantic Segmentation would not be an exception. In this post, we go over one of the most relevant papers on semantic segmentation of general objects - [Deeplab-v3](https://arxiv.org/pdf/1706.05587.pdf). We present an overview of the architecture along with the running Tensorflow implementation and code details for training on the PASCAL VOC 2012 segmentation challenge dataset.

## Semantic Segmentation

Image segmentation is the task of separating an image into multiple segments. Semantic Image Segmentation adds the constraint of classifying each of these segments to their correct classes. When we talk about Semantic Segmentation using DCNNs, ***we are really talking about a classification problem in a pixel-wise level***. In other words, for each pixel in a digital image, the goal is to classify it as one of background, class_1, class_2, … class-n.

<figure>
  <img class="img-responsive center-block" src="{{ site.url }}/assets/deep_segmentation_network/pascal_voc_segmentation_sample.png" alt="ResNet bottleneck layer">
  <figcaption class="caption center">Figure 1: ResNet building block architectures. (Left-most) the original ResNet block. (Right-most) the improved full pre-activation version. </figcaption>
</figure>

In 2014, [Fully Convolutional Networks for Semantic Segmentation](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf), proposed a DCNN based segmentation network that would become a baseline and shift the semantic segmentation field to Deep Learning. In their paper, the authors proposed an encoder-decoder architecture that aimed to bring the success of Deep Leaning in Object Recognition to Semantic Segmentation.

*In an encoder-decoder design pattern, the encoder, usually an object recognition network such as the AlexNet, produces coarse, low level features via a series of convolutions and pooling operations. On the other hand, the decoder network works on reconstructing this signal to a finer representation.*

In FCN, the authors proposed an ImageNet pre-trained VGG-16 net as an encoder and built a decoder model that used some of the intermediate feature layers of the encoder to help in the reconstruction phase of the pipeline. This implementation achieved state-of-the-art performance on the [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/) semantic segmentation challenge with a 20% relative improvement.

Afterwards, many other encoder-decoder architectures emerged. Besides some small changes, most of these architectures provided a slightly different mechanism for reusing intermediate feature layers as a way of improving the segmentation results. As a consequence, we have seen many successful segmentation models in various fields.

There are many challenges that must be overcome in the field of image semantic segmentation. Among them, the native of most deep learning models consist of a series of convolutions intercalated by pooling operation for downsampling. ***Since operations such as max or average pooling are design to reduce the spatial dimensions of a given feature map in detriment of losing some features in the way, segmentation architectures must have a way to deal with the loss of such information in order to output better segmentation maps.***

Different from most encoder-decoder architectures, Deeplab_v3 proposes a different design based on atrous convolutions and *Atrous Spatial Pyramid Pooling* (ASPP) as a way of controlling signal decimation and learning multi-scale contextual features.

## Architecture

To understand the deeplab-v3 architecture, we need to focus on these three components: *(i) the ResNet architecture, (ii) atrous convolution and (iii) ASPP*.

<figure>
  <img class="img-responsive center-block" src="{{ site.url }}/assets/deep_segmentation_network/network_architecture.png" alt="ResNet bottleneck layer">
</figure>

***In summary, the Deeplab-v3 segmentation network takes the ResNet model and adds to it a new block that instead of performing the regular convolutions followed by max pooling, it uses atrous convolutions with different rates to capture multi-scale context. In addition, on top of the last atrous convolution block, it uses Atrous Spatial Pyramid Pooling (ASPP) with different atrous rates as an attempt of classifying regions of an arbitrary scale. Let’s go over each one of the main components.***

### ResNets

ResNet is a very popular DCNN that won the ILSVRC 2015 classification task. One of the main contributions of ResNets was to provide an easy to train framework that could allow the training of very deep DCNNs.

In its original form, ResNets contain 4 computation blocks; each block contains a number of **Residual Units** that perform a series of convolutions in a special way. After each block, a max-pooling operation is performed to reduce spatial dimensions.

The original [ResNet paper](https://arxiv.org/abs/1512.03385), introduces two types of Residual Units, the baseline and the *bottleneck* versions. The baseline Residual Unit is composed by two *3x3* convolutions with Batch Normalization(BN) and RELu.

<figure>
  <img class="img-responsive center-block" src="{{ site.url }}/assets/deep_segmentation_network/resnet_residual_units.png" alt="ResNet bottleneck layer">
  <figcaption class="caption center">Figure 3: ResNet building blocks. (left) baseline; (right) bottleneck unit</figcaption>
</figure>


***A bottleneck Residual Unit consists of a 1×1 layer for reducing dimension, a 3×3 layer, and a 1×1 layer for restoring dimension, which leaves the 3x3 convolution to operate on a less dense feature vector. After, an addition operation between the original input and the result of the non-linear transformation is applied. The connection between the non-linear transformations F(x) and the original input x, provides a way to make the gradient signal from the later layers (where it is probably stronger) to be sent directly to earlier layers, skipping the operations on F(x) - where the gradient might be diminished.***

Although non-bottleneck units also show gain in accuracy as we increase model capacity, the *bottleneck* residual unit has the advantage of performing more computations carrying almost the same number of parameters with similar computational complexity. Therefore, *bottleneck* units are suitable for training deeper models without much concerning for **training time** and **computation resources**.

For the Tensorflow implementation described in this post, we use the **full pre-activation** layer for the ResNet Residual Units. As the name implies, the full pre-activate *bottleneck* layer differs from the original implementation because it applies the activation function (RELu) before convolution.

<figure>
  <img class="img-responsive center-block" src="{{ site.url }}/assets/deep_segmentation_network/various_resnet_based_blocks.png" alt="ResNet bottleneck layer">
  <figcaption class="caption center">Figure 4: ResNet building block architectures. (Left-most) the original ResNet block. (Right-most) the improved full pre-activation version. </figcaption>
</figure>

 Moreover, it makes heavy usage of batch normalization layers before the non-linearities. In [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027), the authors report a performance analysis for these different *bottleneck* architectures and the **full pre-activation** design performed better than the other variations.

 *Note that the only difference among these designs is the order in which BN and RELu are performed.*

### Atrous Convolutions

Atrous (dilated) convolutions are regular convolutions with a factor that allows the convolution filter to act on a larger field of view. Consider a *3x3* regular convolution filter, when the dilation rate is equal to 1, it is equivalent to a standard convolution. If we set the rate factor to 2 though, it has the effect of enlarging the convolution kernel. In practice, it means expanding (dilating) the filter according to the dilation rate while filling the empty spaces with zeros (sparse filter), and then, doing regular convolution.

<figure>
  <img class="img-responsive center-block" src="{{ site.url }}/assets/deep_segmentation_network/atrous_conv.png" alt="ResNet bottleneck layer">
  <figcaption class="caption center">Figure 5: Atrous convolutions with various rates. </figcaption>
</figure>

As can be seen in Figure 4, for a convolution kernel with atrous rate = 2, the filter expands to cover an area equivalent to a *5x5* filter would. However, it acts like a sparse filter where only the original *3x3* cells will do computation and produce output. I said *act* because they are not implemented as sparse filters. Similarly, setting the atrous rate to 3 increases the gap between the kernel’s cells, allowing a regular *3x3* convolution to get signals from a larger, *7x7* equivalent area.

***To summarize, [studies](https://arxiv.org/abs/1704.06857) have suggested that the ability of expanding its filter's field-of-view without increasing memory consumption, makes dilated convolutions very efficient dense feature extractors.***

<figure>
  <img class="img-responsive center-block" src="{{ site.url }}/assets/deep_segmentation_network/extreme_dilation.png" alt="ResNet bottleneck layer">
  <figcaption class="caption center">Figure 6: For a 14x14 input image, a 3x3 filter with dilation rate of 14 makes the atrous convolution behaves like a regular 1x1 convolution.</figcaption>
</figure>

Deeplab also presents experiments regarding the effectiveness of different atrous rates. It is easy to see that, for cases where the atrous rate is very close to the corresponding feature map size, a regular *3x3* atrous filter would act as a standard *1x1* convolution. To prevent this issue, Deeplab_v3 does not implement pooking layer to reduce the feature maps’ spatial dimensions in the atrous convolution blocks. Additionally, understanding that excessive signal decimation is harmful for dense prediction tasks, avoiding extreme downsampling operations in these new blocks minimizes the coarseness of the final output map.

### Atrous Spatial Pyramid Pooling

For ASPP, the idea was to provide more multi-scale context information through atrous convolutions with different rates and also to incorporate global context information to the model by using Global Average Pooling (GAP).

ASPP contains 4 parallel operations; it consists of a *1x1* convolution and three *3x3* convolutions with *rates =(6,12,18)* respectively. Note that at this point, the feature maps, nominal stride is equal to 16. ***The output stride is defined as the ratio of the input image size to the output feature map size. It defines how much of signal decimation the original input signal will suffer as it traverses the network.*** Controlling signal decimation is argued to be very important when segmenting highly detailed images such as aerial imagery.

To incorporate global context information, the authors describe a technique that consists of applying GAP to the last feature map of the model, feed it to a *1x1* convolution with *256 filters* and then, bilinearly upsample the features to the correct dimensions.

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
