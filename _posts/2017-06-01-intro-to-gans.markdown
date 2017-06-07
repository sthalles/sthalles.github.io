---
layout: post
title:  "A Short Introduction to Generative Adversarial Networks"
date:   2017-06-01 10:00:00 -0300
categories: deep learning
short_description: "GANs are a kind of generative model in which two differentiable functions are locked in a game. The generator tries to reproduce data that come from some probability distribution. The discriminator gets to decide if its input comes from the generator or from the true training set."
image_url: "./assets/dcgan/generator-model.png"
---

Image there is this very cool party going on at your neighborhood, you really wanted to be there but there is a problem, to get into the party you need a special ticket that was long sold out. Considering you are especially eager to go to this party, you decide to do something about it. Since the expectations are very high and people are long waiting for the party to happen, the organization decided to hire a very qualified agency for party security. Their primary goal is to not allow anyone to crash the party. To accomplish that, they placed a lot of security guards at the entrance of the party to meticulously check everyone’s tickets for authenticity. As long as you are not any martial artistic master, the only way to get through the security is by fooling them with a fake ticket.
 
There is a big problem with this plan though, you never actually saw how the ticket looks like, so even if you design a ticket based solely on your creativity, you know it’s almost impossible to fool the party security at your first trial. In addition, because you do not want to show off your face to the guards until you have a very decent replica of the ticket, you call out your friend Bob to do the dirty job for you. Bob’s mission is very simple, he will try to get into the party with your fake ticket and if he gets denied, he will come back to you at your office with tips regarding how the ticket should look like. Based on these tips, you make a new version of the ticket and hands it to Bob that goes to do the same. This process keeps repeating until you become able to design a perfect replica capable of fooling the security and therefore, allowing you and your friend Bob inside the party.


<figure>
  <img class="img-responsive center-block" src="{{ site.url }}/assets/dcgan/fake-ticket.jpg" alt="Fake ticket">
  <figcaption class="caption center"> That is a must go party. I actually took that image from a fake ticket generator website! </figcaption>
</figure>

Putting aside the “small holes” on this anecdote, this is how Generative Adversarial Networks (GANs) kind of work. Most of the applications of GANs nowadays are in the domain of computer vision. GANs have been used to address a lot of problems such as training classifiers in a semi-supervised learning manner and to get high resolution images from low resolution ones. This article is intended to provide a quick introduction to GANs as well as to offer a hands on in the problem of generating images.
 
GANs are a kind of generative model designed by [Ian Goodfellow](https://arxiv.org/abs/1406.2661) in which two differentiable functions (represented by neural networks) are locked in a game. The two players, the generator and the discriminator have different roles in this framework. The generator tries to produce data that come from some probability distribution – that would be you trying to reproduce the party’s tickets. The discriminator, on the other hand, gets to decide if its input comes from the generator or from the true training set - that would be the party security comparing your fake ticket with the true ticket and looking for flaws in your design.


<div class="row">
  <div class="col-xs-12 col-sm-6">
    <img class="img-responsive" src="{{ site.url }}/assets/dcgan/svhn-training.gif" alt="SVHN training evolution">
  </div>
  <div class="col-xs-12 col-sm-6">
    <img class="img-responsive" src="{{ site.url }}/assets/dcgan/mnist-training.gif" alt="MNIST training evolution">
  </div>
  <div class="col-xs-12" style="text-align: center;">
    <p class="caption">SVHN's (left) and MNIST (right) generator samples during training.</p>
  </div>
</div>




In summary, the game follows with the generator trying to maximize the probability of making the discriminator mistakes its inputs as real samples while the discriminator is guiding the generator to approximate its samples to the real data distribution. In the perfect equilibrium, the generator would capture the training data distribution which would make the discriminator always unsure whether the generator’s samples are real or not.

Without more ado, let’s dive into the details of the implementation and talk more about GANs as we go. We will discuss the main steps of building a Deep Convolutional Generative Adversarial Network (DCGAN) using Tensorflow based on the original [paper](https://arxiv.org/abs/1511.06434).

<figure>
  <img class="img-responsive" src="{{ site.url }}/assets/dcgan/generator-model.png" alt="Genrator network model">
  <figcaption class="caption center"> Adapted from the DCGAN paper, that is the Generator network described here. Note the not existece of fully connected and pooling layers. </figcaption>
</figure>

The authors of the DCGAN paper describes the combination of some deep learning techniques as key for successfully training GANs. Some of these techniques include, the all convolutional net, which emphasizes the usage of strided convolutions for both, increasing and decreasing the spatial dimensions of the feature map instead of using classical pooling functions, and applying batch normalization i.e. normalizing the input to have zero mean and unit variance in all layers to stabilize learning and to deal with poor weight initialization problems.

### Generator
The network has 4 convolutional layers, all of them followed by batch normalization (except for the output layer) and rectified linear unit (RELU) activations. It takes as input a random vector z (drawn from a normal distribution), which is reshaped to a 4D tensor and start a series of upsampling layers by applying transpose convolutional operations with strides of 2. **Basically, every time we move the convolutional kernel by one pixel on the input feature map, we move it by 2 or more pixels on the output map**. We begin with this very deep but narrow input feature map representation and each transpose convolution operation gives us a wider and shallower feature map. All the transpose convolutions use kernel filters of size 5x5 and the kernel depth goes from 512 all the way down to 3 - representing the RGB color channels. The final layer then outputs a 32x32x3 tensor that will be squashed between -1 and 1 through the [Hyperbolic Tangent](https://reference.wolfram.com/language/ref/Tanh.html) function – which also means that the input data has to be scaled to the interval of -1 to 1.


{% highlight python %}
def generator(z, output_dim, reuse=False, alpha=0.2, training=True):
    """
    Defines the generator network
    :param z: input random vector z
    :param output_dim: output dimension of the network
    :param reuse: Indicates whether or not the existing model variables should be used or recreated
    :param alpha: scalar for lrelu activation function
    :param training: Boolean for controlling the batch normalization statistics
    :return: model's output
    """
    with tf.variable_scope('generator', reuse=reuse):
        fc1 = dense(z, 4*4*512)
        
        # Reshape it to start the convolutional stack
        fc1 = tf.reshape(fc1, (-1, 4, 4, 512))
        fc1 = batch_norm(fc1, training=training)
        fc1 = tf.nn.relu(fc1)
        
        t_conv1 = transpose_conv2d(fc1, 256)
        t_conv1 = batch_norm(t_conv1, training=training)
        t_conv1 = tf.nn.relu(t_conv1)
        
        t_conv2 = transpose_conv2d(t_conv1, 128)
        t_conv2 = batch_norm(t_conv2, training=training)
        t_conv2 = tf.nn.relu(t_conv2)
        
        logits = transpose_conv2d(t_conv2, output_dim)
        
        out = tf.tanh(logits)
        return out
{% endhighlight %}

### Discriminator

The discriminator is also a 4 layer convolutional neural network followed by batch normalization (except its input layer) and [leaky RELU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) activations. Many activation functions will work just fine with this basic GAN architecture, however, leaky RELUs are very popular because they help the gradients to flow more easily through the architecture avoiding the situation where the neurons die due to a possible state in which RELU units always output 0s for all inputs hence, not allowing the gradients to flow back through the network. This caveat is especially important for GANs because **the only way the generator learns is by receiving the gradients from the discriminator**. The network receives a 32x32x3 image tensor and performs regular convolutional operations with ‘same’ padding and strides of 2 - which basically reduces the spatial dimensions (width and height) by half at every layer while it doubles the size of the filters. Finally, the discriminator needs to output probabilities. For that, we use the [Logistic Sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) activation function for the top layer.

{% highlight python %}
def discriminator(x, reuse=False, alpha=0.2, training=True):
    """
    Defines the discriminator network
    :param x: input for network
    :param reuse: Indicates whether or not the existing model variables should be used or recreated
    :param alpha: scalar for lrelu activation function
    :param training: Boolean for controlling the batch normalization statistics
    :return: A tuple of (sigmoid probabilities, logits)
    """
    with tf.variable_scope('discriminator', reuse=reuse):
        # Input layer is 32x32x?
        conv1 = conv2d(x, 64)
        conv1 = lrelu(conv1, alpha)
        
        conv2 = conv2d(conv1, 128)
        conv2 = batch_norm(conv2, training=training)
        conv2 = lrelu(conv2, alpha)
        
        conv3 = conv2d(conv2, 256)
        conv3 = batch_norm(conv3, training=training)
        conv3 = lrelu(conv3, alpha)

        # Flatten it
        flat = tf.reshape(conv3, (-1, 4*4*256))
        logits = dense(flat, 1)

        out = tf.sigmoid(logits)
        return out, logits
{% endhighlight %}

Note that in this framework, the discriminator acts as a regular binary classifier. Half of the time it receives images from the training set and the other half from the generator. Back to our adventure, to successfully reproduce the party’s ticket, the only source of information you had was the feedback from our friend Bob. Put differently, the quality of the information Bob provides to you at each trial was essential to get the job done. ***Equivalently, every time the discriminator notices a difference between the real and fake images, the gradients that flow from the discriminator to the generator allows for the generator to adjust its parameters to approximate its sample outputs to the ones of the training set***. This is how important the discriminator is. In fact, ***the generator will be as good as producing data as the discriminator is at telling them apart***.
 
### Losses

Now, let’s describe what I believe is the trickiest part of this architecture, the losses. First, we already know that the discriminator receives images from both, the training set and from the generator. We want the discriminator to be able to distinguish between real and fake images. Every time we run a mini-batch through the discriminator, we get logits, i.e. the unscaled values from the model. However, if we think about it, we can divide the mini-batches that the discriminator receives in two types, one composed only with real images that come from the training set and another one with only fake images a.k.a. the ones created by the generator.

{% highlight python %}
def model_loss(input_real, input_z, output_dim, alpha=0.2, smooth=0.1):
    """
    Get the loss for the discriminator and generator
    :param input_real: Images from the real dataset
    :param input_z: random vector z
    :param out_channel_dim: The number of channels in the output image
    :param smooth: label smothing scalar 
    :return: A tuple of (discriminator loss, generator loss)
    """
    g_model = generator(input_z, output_dim, alpha=alpha)
    d_model_real, d_logits_real = discriminator(input_real, alpha=alpha)
        
    d_model_fake, d_logits_fake = discriminator(g_model, reuse=True, alpha=alpha)
    
    # for the real images, we want them to be classified as positives,  
    # so we want their labels to be all ones. 
    # notice here we use label smoothing for helping the discriminator to generalize better. 
    # Label smoothing works by avoiding the classifier to make extreme predictions when extrapolating.
    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, labels=tf.ones_like(d_logits_real) * (1 - smooth)))
    
    # for the fake images produced by the generator, we want the discriminator to clissify them as false images,
    # so we set their labels to be all zeros.
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.zeros_like(d_model_fake)))
    
    # since the generator wants the discriminator to output 1s for its images, it uses the discriminator logits for the
    # fake images and assign labels of 1s to them.
    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.ones_like(d_model_fake)))
    
    d_loss = d_loss_real + d_loss_fake
    
    return d_loss, g_loss
{% endhighlight %}

Because the generator and the discriminator networks train simultaneity, GANs require two optimizers to run at the same time. Each one for minimizing the discriminator and generator’s loss functions respectively. Since we want the discriminator to output probabilities close to 1 for real images and near 0 for fake images, we need two partial losses for the discriminator. The total loss for the discriminator is then, the sum of the two partial losses - one for maximizing the probabilities for the real images and another for minimizing the probability of fake images.

<figure>
  <img class="img-responsive center-block" src="{{ site.url }}/assets/dcgan/svhn-real-generated.png" alt="Comparing real and fake svhn images">
  <figcaption class="caption center"> Comparing real (left) and generated (right) SVHN sample images. Although some images look blured while some others are not recognizable, its noticible that the data structure was well captured by the model. </figcaption>
</figure>

In the beginning of training, these two types of batches look very distinct from one another, which makes the discriminator’s job easy. However, because the generator loss is set to maximize the probability of the fake data, in other words, it trains to output images from the same probability distribution as the training set images; as training goes, the generator starts to output images that look closer to the images from the training set. Hence, at each training step, the two types of mini-batches begin looking similar and similar (in structure) to one another, and as a result, it makes the discriminator more and more uncertain about what images are the real and fake ones.

<figure>
  <img class="img-responsive" src="{{ site.url }}/assets/dcgan/mnist-real-generated.png" alt="Comparing real and fake mnist images">
  <figcaption class="caption center"> Comparing real (left) and generated (right) MNIST sample images. Because MNIST images have a simpler data structure, the model was able to produce more realistic samples when compared to the SVHNs. </figcaption>
</figure>

### Concluding

GANs are, with no doubt, one of the hottest subjects in machine learning right now. Mainly because it has been treated as key to unlock some **unsupervised learning** methods that would make these models even more powerful. 
 
There has been a lot of advancements in training GANs to achieve state-of-the-art results, some of them can be found at [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498), where the authors describe some advanced techniques for both image generation and unsupervised learning training with GANs.
 
In addition, there are two very good blog posts that really inspired me when learning to understand how GANs function: [Generative Models](https://blog.openai.com/generative-models/#gan) and [Generative Adversarial Networks Explained with a Classic Spongebob Squarepants Episode](https://medium.com/@awjuliani/generative-adversarial-networks-explained-with-a-classic-spongebob-squarepants-episode-54deab2fce39).
 
Enjoy!