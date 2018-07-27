---
layout: post
title:  "Advanced GANs - Normalization techniques for Generator and Discriminator training."
date:   2018-08-30 08:00:00 -0300
categories: deep learning
short_description: "..."
image_url: "/assets/serving_tf_models/tf_serving_cover.jpeg"
---

## Introduction

Lately, Generative Models are drawing a lot of attention. Much of that comes from Generative Adversarial Networks (GANs). Invented by Goodfellow [REFERENCE], GANs are a framework in which two players compete with one another. The two actors, the generator G and discriminator D are both represented by function approximators. Moreover, they play different roles in the game.

Given a training data D<sub>t</sub>, the generator creates samples as an attempt to mimic the ones from the same probability distribution as D<sub>t</sub>.

The discriminator, on the other hand, is a common binary classifier. It has two main jobs. First, it judges whether its received input comes from the true data distribution (D<sub>t</sub>) or from Generator distribution. In addition, D also guides G to create more realistic samples by passing to G its gradients.

In this game, G takes random noise as input and generates a sample image G<sub>sample</sub>. This sample is designed to maximize the probability of making D mistakes it as coming from real training set D<sub>t</sub>.  

During training, D receives half of the time images from the training set, and the other half, images from the generator network - G<sub>sample</sub>. D is trained to maximize the probability of assigning the correct class label to both: real samples and fake samples (from G). In the end, the hope is that the game finds an equilibrium - the Nash equilibrium.

In this situation, the Generator would capture the data probability distribution. And the discriminator, in turn, would not be able to distinguish between real or fake samples and outputs probability 1/2 everywhere.

GANs have been used in a lot of different applications in the past few years. Some of them include: generating synthetic data, Image in-paining, semi-supervised learning, super-resolution, text to image generation and more.

However, much of the recent work on GANs is focused on developing techniques to stabilize training. In fact, GANs are known to be unstable (during training) and very sensitive to the choice of hyper-parameters.

In this context, this piece presents an overview of two of the most relevant works for improving GANs. Specifically, here, we describe recent methods for improving the quality of the Generator's samples. To do that, we address 3 techniques explored in the recent paper [REFERENCE].

All the code developed with the Tensorflow Eager execution API is available at: [CODE].

## Convolutional GANs

Image Generation GANs started to work after the introduction of the [DCGAN PAPER]. In the DCGAN model, the generator G is composed as a series of transpose convolution operations. These ops, take in a random noise vector z and transforms it by progressively increasing its spatial dimensions while decreasing its feature volume depth.

Also, the DCGAN uses Batch Normalization (BN) as a keystone in their architecture. Batch Norm works by normalizing the input features of a layer to have zero mean and unit variance. They found that BN was essential for getting Deeper models to work without falling into mode collapse. Mode collapse is the situation in which G creates samples with very low diversity. In other words, G returns the same looking samples for different input signals. Also, BN helps to deal with poor initialization problems.

In addition, DCGAN uses Leaky ReLU activations in the discriminator network. Different from the regular ReLU function, Leaky ReLU allows the pass of a small gradient signal for negative values. In this way, it makes the gradients from the discriminator flows stronger into the generator. Instead of passing a gradient (slope) of 0 in the back-prop pass, it passes a small negative gradient.

After the introduction of DCGANs, very few things changed in the design architecture of Image Generation GANs. The building blocks: Transpose Convolutions and Batch Normalization are still there in most architectures. However, much of the new work focuses on how to make the GAN training more stable.

## Self-Attention GANs

The SAGAN paper introduces the concept of Self Attention for Generative Adversarial Networks. Attention techniques have been explored, with success, in other problems such as Machine Translation. Here, the key idea is to make the generator able to create samples with global detailing information.

We know that regular GANs are heavily based on convolutions. These operations use a local receptive field (the convolutional kernel) to learn representations. Convolutions have very nice properties such as parameter sharing and translation invariance.

A typical Deep ConvNet learns representations in a hierarchical fashion. In other words, for a regular image classification ConvNet, simple features like edges and corner are learned in the first few layers. However, ConvNets use these simple representations to learn more complicated ones. In short, ConvNets learn representations that are expressed in terms of simpler representations.

As a result, long-range dependency might be hard to learn. Indeed, it might only be possible for very decimated feature vectors. The problem is that, at this point, the amount of signal loss is such that it becomes difficult to model long-range details.

Take a look at these images. They are from the DCGAN model trained on ImageNet. As pointed in the paper, most of the image content that does not exhibit elaborated shapes looks ok. Put it differently, GANs usually do not have problems modeling less structural content like the sky or the ocean. Nonetheless, the task of creating geometrically complex forms, such as four-legged animals, is far more challenging. That is because, complicated geometrical contours demand long-range detail that the convolution ope, by itself, cannot grasp. That is where attention comes into play.

The idea is to give to the generator, information from a broader feature space. Not only the convolutional kernel range. By doing so, the generator can create samples with fine-detail resolution.

### Implementation

Given the output features of a given convolutional layer, the first step is to transform these features in 3 different representations. Using 1x1 convolutions, we get three feature spaces: f, g, and h. Here, f and g are used to calculate the attention. To do that, we linearly combine f and g using a matrix multiplication and the result is fed into a softmax layer.

The resulting tensor is linearly-combined with h and finally scaled by gamma. Note that gamma starts as 0. At the beginning of training, gamma zeros out the attention layers. Consequently, the network only relies on local representations (from the regular convolutional layers). However, as gamma receives gradient descent updates, the more the network allows the passing of signals from non-local fields.

Also, note that the feature vectors f and g have different dimensions than h. As a matter of fact, f and g use 8 times less convolutional filters than h does.

## Spectral Normalization

Another important contribution of  [REFERENCE], is the use of spectral normalization in the generator. Previously, [REFERECE SN] proposed restricting the spectral norm of each layer of the discriminator. In a few words, spectral normalization constrains the Lipschitz constant of the discriminator, which in practice, worked very well.

However, doing the same for the generator network proved to be very effective for improving GANs performance. For G, spectral norm prevents the parameters to get very big and avoids unwanted gradients.

Yet, there is one fundamental problem when training a normalized discriminator. Previous work has shown that regularization of the discriminator makes the GAN training slower. For this reason, some workarounds consist of updating the discriminator a few times, before updating the generator. For instance, regularized discriminators might require 5 or more update steps for 1 generator update.

Nevertheless, the paper authors used a simple technique to solve this problem. It consists of providing different learning rates for optimizing the generator and discriminator. Imbalanced learning rates for G and D were first introduced by [REFERENCE].

In fact, D trains with a learning rate 3 times greater than G - 0.003 and 0.001 respectively. A larger learning rate means that the discriminator will absorb a larger part of the gradient signal. Hence, a higher learning rate alleviates on the problem of slow learning of the regularized discriminator.

<figure>
  <img class="img-responsive center-block" src="{{ site.url }}/assets/advanced_gans/self_attention_module.png" alt="ResNet bottleneck layer">
</figure>


{% gist 507ce723226274db8097c24c5359d88a %}

## Exporting a Model for Serving

The first step to serve an ML model built in TensorFlow is to make sure it is in the right format. To do that, TensorFlow provides the [SavedModel](https://www.tensorflow.org/programmers_guide/saved_model) class.

```SavedModel is the universal serialization format for TensorFlow models.```

If you are familiar with TF, you have probably used the TensorFlow Saver to persist your model’s variables.

The TensorFlow Saver provides functionalities to save/restore the model’s checkpoint files to/from disk. In fact, SavedModel wraps the TensorFlow Saver and it is meant to be the standard way of exporting TF models for serving.

The SavedModel object has some nice features.

First, it lets you save more than one meta-graph to a single SavedModel object. In other words, it allows us to have different graphs for different tasks.

For instance, suppose you just finished training your model. In most situations, to perform inference, your graph doesn’t need some training-specific operations. These ops might include the optimizer’s variables, learning rate scheduling tensors, extra pre-processing ops, and so on.

Moreover, you might want to serve a quantized version of a graph for mobile deployment.



In this context, SavedModel allows you to save graphs with different configurations. In our example, we would have three different graphs with corresponding tags such as ```“training”, “inference”, and “mobile”```. Also, these three graphs would all share the same set of variables — which emphasizes memory efficiency.

Not so long ago, when we wanted to deploy TF models on mobile devices, we needed to know the names of the input and output tensors for feeding and getting data to/from the model. This need forced programmers to search for the tensor they needed among all tensors of the graph. If the tensors were not properly named, the task could be very tedious.

To make things easier, SavedModel offers support for [SignatureDefs](https://www.tensorflow.org/serving/signature_defs). ```In summary, SignatureDefs define the signature of a computation supported by TensorFlow```. It determines the proper input and output tensors for a computational graph. Simply put, with these signatures you can specify the exact nodes to use for input and output.

To use its built-in serving APIs, TF Serving requires models to include one or more SignatureDefs.

<figure>
  <img class="img-responsive center-block" src="{{ site.url }}/assets/serving_tf_models/signature_overview.png" alt="ResNet bottleneck layer">
</figure>

To create such signatures, we need to provide definitions for inputs, outputs, and the desired method name. The definitions of ```inputs``` and ```Outputs``` represent a mapping from string to ```TensorInfo objects``` (more on this latter). Here, we define the default tensors for feeding and receiving data to and from a graph. The ```method_name``` parameter targets one of the TF high-level serving APIs.

Currently, there are three serving APIs: ```Classification, Predict, and Regression```. Each signature definition matches a specific RPC API. The Classification SegnatureDef is used for the Classify RPC API. The Predict SegnatureDef is used for the Predict RPC API, and on.

For the Classification signature, there must be an inputs tensor (to receive data) and at least one of two possible output tensors: classes and/or scores. The Regression SignatureDef requires exactly one tensor for input and another for output. Lastly, the Predict signature allows for a dynamic number of input and output tensors.

In addition, SavedModel supports assets storage for cases where ops initialization depends on external files. Also, it has mechanisms for clearing devices before creating the SavedModel.

Now, let’s see how can we do it in practice.

## Setting up the Environment

Before we begin, clone this [TensorFlow DeepLab-v3 implementation](https://github.com/sthalles/deeplab_v3) from Github.

DeepLab is Google’s best semantic segmentation ConvNet. Basically, the network takes an image as input and outputs a mask-like image that separates certain objects from the background.

This version was trained on the [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) segmentation dataset. Thus, it can segment and recognize up to 20 classes. If you want to know more about Semantic Segmentation and DeepLab-v3, take a look at [Diving into Deep Convolutional Semantic Segmentation Networks and Deeplab_V3](https://medium.freecodecamp.org/diving-into-deep-convolutional-semantic-segmentation-networks-and-deeplab-v3-4f094fa387df).

All the files related to serving reside into: ```./deeplab_v3/serving/```. There, you will find two important files: [deeplab_saved_model.py](https://github.com/sthalles/deeplab_v3/blob/master/serving/deeplab_saved_model.py) and [deeplab_client.ipynb](https://github.com/sthalles/deeplab_v3/blob/master/serving/deeplab_client.ipynb)
Before going further, make sure to download the Deeplab-v3 pre-trained model. Head to the GitHub repository above, click on the checkpoints link, and download the folder named ```16645/```.

In the end, you should have a folder named ```tboard_logs/``` with the ```16645/``` folder placed inside it.
''
<figure>
  <img class="img-responsive center-block" src="{{ site.url }}/assets/serving_tf_models/setup_overview.png" alt="ResNet bottleneck layer">
</figure>

Now, we need to create two Python virtual environments. One for Python 3 and another for Python 2. For each env, make sure to install the necessary dependencies. You can find them in the [serving_requirements.txt](https://github.com/sthalles/deeplab_v3/blob/master/serving/serving_requirements.txt) and [client_requirements.txt](https://github.com/sthalles/deeplab_v3/blob/master/serving/client_requirements.txt) files.

We need two Python envs because our model, DeepLab-v3, was developed under Python 3. However, the TensorFlow Serving Python API is only published for Python 2. Therefore, to export the model and run TF serving, we use the Python 3 env. For running the client code using the TF Serving python API, we use the PIP package (only available for Python 2).

Note that you can forgo the Python 2 env by using the Serving APIs from bazel. Refer to the [TF Serving Instalation](https://www.tensorflow.org/serving/setup#aptget) for more details.

With that step complete, let’s start with what really matters.

## How to do it

To use SavedModel, TensorFlow provides an easy to use high-level utility class called [SavedModelBuilder](https://www.tensorflow.org/api_docs/python/tf/saved_model/builder/SavedModelBuilder). The SavedModelBuilder class provides functionalities to save multiple meta graphs, associated variables, and assets.

Let’s go through a running example of how to export a Deep Segmentation CNN model for serving.

As mentioned above, to export the model, we use the SavedModelBuilder class. It will generate a SavedModel protocol buffer file along with the model’s variables and assets (if necessary).

Let’s dissect the code.

{% gist bc6e5ae4e12f1ad57b30f7e08a70aa87 %}

The SavedModelBuilder receives (as input) the directory where to save the model’s data. Here, the ```export_path``` variable is the concatenation of ```export_path_base``` and the ```model_version```. As a result, different model versions will be saved in separate directories inside the ```export_path_base``` folder.

Let’s say we have a baseline version of our model in production, but we want to deploy a new version of it. We have improved our model’s accuracy and want to offer this new version to our clients.

To export a different version of the same graph, we can just set ```FLAGS.model_version``` to a higher integer value. Then a different folder (holding the new version of our model) will be created inside the ```export_path_base``` folder.

Now, we need to specify the input and output Tensors of our model. To do that, we use SignatureDefs. Signatures define what type of model we want to export. It provides a mapping from strings (logical Tensor names) to TensorInfo objects. ```The idea is that, instead of referencing the actual tensor names for input/output, clients can refer to the logical names defined by the signatures.```

For serving a Semantic Segmentation CNN, we are going to create a Predict Signature. Note that the *build_signature_def()* function takes the mapping for input and output tensors as well as the desired API.

A SignatureDef requires specification of: ```inputs, outputs, and method name```. Note that we expect three values for inputs — an image, and two more tensors specifying its dimensions (height and width). For the outputs, we defined just one outcome — the segmentation output mask.

{% gist e2fc52b75c60f8cb5fcf9ba2d88e9e9f %}

Note that the strings ‘image’, ‘height’, ‘width’ and ‘segmentation_map’ are not tensors. ```Instead, they are logical names that refer to the actual tensors input_tensor, image_height_tensor, and image_width_tensor```. Thus, they can be any unique string you like.

Also, the mappings in the SignatureDefs relates to TensorInfo protobuf objects, not actual tensors. To create TensorInfo objects, we use the utility function: [tf.saved_model.utils.build_tensor_info(tensor)](https://www.tensorflow.org/api_docs/python/tf/saved_model/utils/build_tensor_info).

That is it. Now we call the *add_meta_graph_and_variables()* function to build the SavedModel protocol buffer object . Then we run the *save()* method and it will persist a snapshot of our model to the disk containing the model’s variables and assets.

{% gist 7130ce6e1811630c86634a38f0448ce3 %}

We can now run [deeplab_saved_model.py](https://github.com/sthalles/deeplab_v3/blob/master/serving/deeplab_saved_model.py) to export our model.

If everything went well you will see the folder ```./serving/versions/1```. Note that the ‘1’ represents the current version of the model. Inside each version sub-directory, you will see the following files:

<figure>
  <img class="img-responsive center-block" src="{{ site.url }}/assets/serving_tf_models/exported_model_overview.png" alt="ResNet bottleneck layer">
</figure>

- ***saved_model.pb*** or ***saved_model.pbtxt***. This is the serialized SavedModel file. It includes one or more graph definitions of the model, as well as the signature definitions.

- variables. This folder contains the serialized variables of the graphs.

Now, we are ready to launching our model server. To do that, run:

{% highlight Bash shell scripts %}
$ tensorflow_model_server --port=9000 --model_name=deeplab --model_base_path=<full/path/to/serving/versions/>
{% endhighlight %}

The *model_base_path* refers to where the exported model was saved. Also, we do not specify the version folder in the path. The model versioning control is handled by TF Serving.

### Generating Client Requests

The client code is very straightforward. Take a look at it in: [deeplab_client.ipynb](https://github.com/sthalles/deeplab_v3/blob/master/serving/deeplab_client.ipynb).

First, we read the image we want to send to the server and convert it to the right format.

Next, we create a gRPC stub. The stub allows us to call the remote server’s methods. To do that, we instantiate the ```beta_create_PredictionService_stub``` class of the ```prediction_service_pb2``` module. At this point, the stub holds the necessary logic for calling remote procedures (from the server) as if they were local.

Now, we need to create and set the request object.

Since our server implements the TensorFlow Predict API, we need to parse a Predict request. To issue a Predict request, first, we instantiate the ```PredictRequest``` class from the ```predict_pb2``` module. We also need to specify the ```model_spec.name``` and ```model_spec.signature_name``` parameters. The name param is the ‘model_name’ argument that we defined when we launched the server. And the signature_name refers to the logical name assigned to the *signature_def_map()* parameter of the *add_meta_graph()* routine.

{% gist d491292e96800b74fa9bde6e40167cf3 %}

Next, we must supply the input data as defined in the server’s signature. Remember that, in the server, we defined a Predict API to expect an image as well as two scalars (the image’s height and width). To feed the input data into the request object, TensorFlow provides the utility *tf.make_tensor_proto()*. This method creates a TensorProto object from a numpy/Python object. We can use it to feed the image and its dimensions to the request object.

Looks like we are ready to call the server. To do that, we call the *Predict()* method (using the stub) and pass the request object as an argument.

For requests that return a single response, gRPC supports both: synchronous and asynchronous calls. Thus, if you want to do some work while the request is being processed, we could call *Predict.future()* instead of *Predict()*.

{% gist cc1e58bdfebad051f8e1fa253dfae2f7 %}

Now we can fetch and enjoy the results.

<figure>
  <img class="img-responsive center-block" src="{{ site.url }}/assets/serving_tf_models/serving_results.png" alt="ResNet bottleneck layer">
</figure>

Hope you liked this article. Thanks for reading!
