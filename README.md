# Automatic Image Captioning Using CNNs

# Chapter I

## 1. Introduction

Caption generation is a challenging artificial intelligence problem where a textual description must be generated for a given photograph. Image captioning, i.e., describing the content observed in an image, has received a significant amount of attention in recent years. It requires both methods from computer vision to understand the content of the image and a language model from the field of natural language processing to turn the understanding of the image into words in the right order. Recently, deep learning methods have achieved state-of-the-art results on examples of this problem. 
Image captioning has many potential applications in real life. A noteworthy one would be to save the captions of an image so that it can be retrieved easily at a later stage just on the basis of this description. It is applicable in various other scenarios, e.g., recommendation in editing applications, usage in virtual assistants, for image indexing, and support of the disabled. With the availability of large datasets, deep neural network (DNN) based methods have been shown to achieve impressive results on image captioning tasks. These techniques are largely based on recurrent neural nets (RNNs), often powered by a Long-Short-Term-Memory (LSTM) component. LSTM nets have been considered as the de-facto standard for vision-language tasks of image captioning [1], visual question answering [2], question generation [3], and visual dialog [4], due to their compelling ability to memorise long-term dependencies through a memory cell. However, the complex addressing and overwriting mechanism combined with inherently sequential processing, and significant storage required due to back-propagation through time (BPaTT), poses challenges during training. Also, in contrast to CNNs, that are non-sequential, LSTMs often require more careful engineering, when considering a novel task. Previously, CNNs have not matched up to the LSTM performance on vision-language tasks. In this project I have used CNNs and LSTMs to serve the purpose of the Image Captioning and achieve decent accuracy.  Figure shown below can be used to understand the task of Image Captioning in a detailed manner. 




<img src="Image/Sample IC.jpg" style="width:800px;height:300px;">



This figure would be labelled by different people as the following sentences : 

A man and a girl sit on the ground and eat .
A man and a little girl are sitting on a sidewalk near a blue bag and eating .
A man wearing a black shirt and a little girl wearing an orange dress share a treat .

But when it comes to machines, automatically generating this textual description from an artificial system is what is called Image Captioning. The task is straightforward – the generated output is expected to describe in a single sentence what is shown in the image – the objects present, their properties, the actions being performed and the interaction between the objects, etc. But to replicate this behaviour in an artificial system is a huge task, as with any other image processing problem and hence the use of complex and advanced techniques such as Deep Learning to solve the task.


## 2. Motivation

Generating captions for images is a vital task relevant to the area of both **Computer Vision** and **Natural Language Processing**. Mimicking the human ability of providing descriptions for images by a machine is itself a remarkable step along the line of Artificial Intelligence. The main challenge of this task is to capture how objects relate to each other in the image and to express them in a natural language (like English).Traditionally, computer systems have been using pre-defined templates for generating text descriptions for images. However, this approach does not provide sufficient variety required for generating lexically rich text descriptions. This shortcoming has been suppressed with the increased efficiency of neural networks. Many state of art models use neural networks for generating captions by taking image as input and predicting next lexical unit in the output sentence. Some real world scenarios where Image Captioning plays a vital role are as follows :
* 1. Self driving cars — Automatic driving is one of the biggest challenges and if we can properly caption the scene around the car, it can give a boost to the self driving system.
* 2. Aid to the blind — We can create a product for the blind which will guide them travelling on the roads without the support of anyone else. We can do this by first converting the scene into text and then the text to voice. Both are now famous applications of Deep Learning.
* 3. CCTV cameras are everywhere today, but along with viewing the world, if we can also generate relevant captions, then we can raise alarms as soon as there is some malicious activity going on somewhere. This could probably help reduce some crime and/or accidents.
* 4. Automatic Captioning can help, make Google Image Search as good as Google Search, as then every image could be first converted into a caption and then search can be performed based on the caption.

## 3. Objective
This project aims at generating captions for images using neural language models. There has been a substantial increase in number of proposed models for image captioning task since neural language models and convolutional neural networks(CNN) became popular. This project has its base on one of such works, which uses a variant of Recurrent neural network coupled with a CNN. I intend to enhance this model by making subtle changes to the architecture and using phrases as elementary units instead of words, which may lead to better semantic and syntactical captions. RNN’s have become very powerful. Especially for sequential data modelling. Image captioning is an application of one to many type of RNNs. For a given input image model predicts the caption based on the vocabulary of train data. I have considered the Flickr8k dataset for this project.

## 4. Summary 
Image captioning is an important task, applicable to virtual assistants, editing tools, image indexing, and support of the disabled. In recent years significant progress has been made in image captioning, using Recurrent Neural Networks powered by long-short-term-memory (LSTM) units. Despite mitigating the vanishing gradient problem, and despite their compelling ability to memorize dependencies, LSTM units are complex and inherently sequential across time. To address this issue, recent work has shown benefits of convolutional networks for machine translation and conditional image generation. The task of image captioning can be divided into two modules logically – one is an image based model – which extracts the features and nuances out of our image, and the other is a language based model – which translates the features and objects given by our image based model to a natural sentence. Chapter 2 explains the detailed methodology of this project including Data Collection, Data Cleaning, Loading the training set, Data Preprocessing — Images, Data Preprocessing — Captions, Word Embeddings and the Model Architecture. Chapter 3 discusses about the results obtained and the inferences drawn from them. This chapter lists down the prediction made by the model on test data i.e. the captions generated given the test image. The chapter also lists down instances when the model is not able to generate relevant captions from the image. Chapter 4 concludes the project and discusses the drawbacks and future scope of the same. We must understand that the images used for testing must be semantically related to those used for training the model. For example, if the model is trained on images of cats, dogs, etc. then it should not be tested or used to generate captions for images of air planes, waterfalls, etc. This is an example where the distribution of the train and test sets will be very different and in such cases no Machine Learning model would give good performance.

# Chapter II

## Data Collection 

There are many open source datasets available for this problem, like Flickr 8k [5] (containing8k images), Flickr 30k [6] (containing 30k images), MS COCO [7] (containing 180k images), etc. But a good dataset to use when getting started with image captioning is the Flickr8K dataset. The reason is because it is realistic and relatively small so that we can download it and build models on our workstation using a CPU. Flickr8k is a labeled dataset consisting of 8000 photos with 5 captions for each photos. It includes images obtained from the Flickr website. Another advantage of using Flickr8k is that data is properly labelled. For each image 5 captions have been provided. The images were chosen from six different Flickr groups, and tend not to contain any well-known people or locations, but were manually selected to depict a variety of scenes and situations.  
The images in this dataset are bifurcated as follows:

	Training Set — 6000 images
	Validation Set — 1000 images
	Test Set — 1000 images

## Understanding the Data 

The Flickr8k dataset also consists of some text files included as part of the dataset. One of the files is the “Flickr8k.token.txt” which contains the name of each image along with the 5 captions. Thus every line contains the <Image name>#i <Caption>, (where 0≤i≤4 ) i.e. the name of the image, caption number (0 to 4) and the actual caption. Table 1 shows the format in which data is given in this text file. 
