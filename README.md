# Automatic Image Captioning Using CNNs

# Chapter I

## 1. Introduction

Caption generation is a challenging artificial intelligence problem where a textual description must be generated for a given photograph. Image captioning, i.e., describing the content observed in an image, has received a significant amount of attention in recent years. It requires both methods from computer vision to understand the content of the image and a language model from the field of natural language processing to turn the understanding of the image into words in the right order. Recently, deep learning methods have achieved state-of-the-art results on examples of this problem. 
Image captioning has many potential applications in real life. A noteworthy one would be to save the captions of an image so that it can be retrieved easily at a later stage just on the basis of this description. It is applicable in various other scenarios, e.g., recommendation in editing applications, usage in virtual assistants, for image indexing, and support of the disabled. With the availability of large datasets, deep neural network (DNN) based methods have been shown to achieve impressive results on image captioning tasks. These techniques are largely based on recurrent neural nets (RNNs), often powered by a Long-Short-Term-Memory (LSTM) component. LSTM nets have been considered as the de-facto standard for vision-language tasks of image captioning [1], visual question answering [2], question generation [3], and visual dialog [4], due to their compelling ability to memorise long-term dependencies through a memory cell. However, the complex addressing and overwriting mechanism combined with inherently sequential processing, and significant storage required due to back-propagation through time (BPaTT), poses challenges during training. Also, in contrast to CNNs, that are non-sequential, LSTMs often require more careful engineering, when considering a novel task. Previously, CNNs have not matched up to the LSTM performance on vision-language tasks. In this project I have used CNNs and LSTMs to serve the purpose of the Image Captioning and achieve decent accuracy.  Figure shown below can be used to understand the task of Image Captioning in a detailed manner. 




<img src="Image/Sample IC.jpg" style="width:1000px;height:500px;">



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

## 1. Data Collection 

There are many open source datasets available for this problem, like Flickr 8k [5] (containing8k images), Flickr 30k [6] (containing 30k images), MS COCO [7] (containing 180k images), etc. But a good dataset to use when getting started with image captioning is the Flickr8K dataset. The reason is because it is realistic and relatively small so that we can download it and build models on our workstation using a CPU. Flickr8k is a labeled dataset consisting of 8000 photos with 5 captions for each photos. It includes images obtained from the Flickr website. Another advantage of using Flickr8k is that data is properly labelled. For each image 5 captions have been provided. The images were chosen from six different Flickr groups, and tend not to contain any well-known people or locations, but were manually selected to depict a variety of scenes and situations.  
The images in this dataset are bifurcated as follows:

	Training Set — 6000 images
	Validation Set — 1000 images
	Test Set — 1000 images

## 2.Understanding the Data 

The Flickr8k dataset also consists of some text files included as part of the dataset. One of the files is the “Flickr8k.token.txt” which contains the name of each image along with the 5 captions. Thus every line contains the <Image name>#i <Caption>, (where 0≤i≤4 ) i.e. the name of the image, caption number (0 to 4) and the actual caption. Table 1 shows the format in which data is given in this text file.
	
<img src="Image/Flickr8k.token.txt Sample .png" style="width:800px;height:300px;">
	
## 3. Data Cleaning 

When we deal with text, we generally perform some basic cleaning like lower-casing all the words (otherwise“hello” and “Hello” will be regarded as two separate words), removing special tokens (like ‘%’, ‘$’, ‘#’, etc.), eliminating words which contain numbers (like ‘hey199’, etc.).  In  this project, while text cleaning : 

* Stop words have not been removed because if we don’t teach our model how to insert stop words like a, an, the, etc , it would not generate correct english.
* Stemming has not been performed because if we feed in stemmed words, the model is also going to learn those stemmed words . So for example, if the word is ‘running’ and we stem it and make it ‘run’ , the model will predict sentences like “Dog is run” instead of “Dog is running”. 
* All the text has been converted to lower case so that ‘the’ and ‘The’ are treated as the same words. 
* Numbers, Punctuations and special symbols like ‘@‘, ‘#’ and so on have been removed, so that we generate sentences without any punctuation or symbols. This is beneficial as it helps to reduce the vocabulary size. Small vocabulary size means less number of neurons and hence less parameters to be computed and hence less overfitting. 

### 3.1 Vocabulary Creation 

Next step is to create a vocabulary of all the unique words present across all the 8000*5 (i.e. 40000) image captions in the dataset. To make the vocabulary set, a python inbuilt data structure called **SET** has been used. Set is used to store all the unique words in the dataset. Total unique words that are there in the dataset are 8424. However, many of these words will occur very few times , say 1, 2 or 3 times. Since it is a predictive model, we would not like to have all the words present in our vocabulary but the words which are more likely to occur or which are common. This helps the model become more robust to outliers and make less mistakes. Hence a threshold has been chosen and if the frequency of the word is less than the threshold frequency, then that particular word has been removed from our vocabulary set. Finally for storing the words and their corresponding frequencies, another python data structure called **Counter** has been used. 

**Note : A counter is a subclass of dict. Therefore it is an unordered collection where elements and their respective count are stored as a dictionary. It is imported from the collections module.**

*After applying the frequency threshold filter, we get the total vocabulary size as 1652 words (having frequency more than 10).*

### 3.2 Loading the Training Set 

The dataset also includes “Flickr_8k.trainImages.txt” file which contains the name of the images (or image ids) that belong to the training set. So all the training image ids have been mapped with the captions and stored in a dictionary. Another important step while creating the dictionary was to add a **‘startseq’ and ‘endseq’** token in every caption, since RNN or LSTM based layers have been used for generating text. In such layers, the generation of text takes place such that the output of a previous unit acts as an input to the next unit. So such model can keep generating words for infinite time steps. Hence we need to specify a way which tells the model to stop generating words further. This is accomplished by adding two tokens in the captions i.e. 

* ‘startseq’ -> This is a start sequence token which was added at the start of every caption.
* ‘endseq’ -> This is an end sequence token which was added at the end of every caption.

The model will be able to generate this token only if we have it in our training set. 

## 4. Data Pre-processing : Images 

Images are nothing but input (X) to our model. Any input to a model must be given in the form of a vector. Hence all the images have to be converted into a fixed size vector which can then be fed as input to a Neural Network. For this purpose, transfer learning has been used. 

### 4.1 Transfer Learning 

Transfer learning (TL) is a research problem in machine learning (ML) that focuses on storing knowledge gained while solving one problem and applying it to a different but related problem [8]. For example, knowledge gained while learning to recognize cars could apply when trying to recognize trucks. Transfer learning is popular in deep learning given the enormous resources required to train deep learning models or the large and challenging datasets on which deep learning models are trained. In transfer learning, we first train a base network on a base dataset and task, and then we repurpose the learned features, or transfer them, to a second target network to be trained on a target dataset and task. This process will tend to work if the features are general, meaning suitable to both base and target tasks, instead of specific to the base task. 

#### 4.1.1 Pre-Training

When we train the network on a large dataset(for example: ImageNet) , we train all the parameters of the neural network and therefore the model is learned. It may take hours on your GPU. 

#### 4.1.2  Fine Tuning

We can give the new dataset to fine tune the pre-trained CNN. Consider that the new dataset is almost similar to the original dataset used for pre-training. Since the new dataset is similar, the same weights can be used for extracting the features from the new dataset. 

When the new dataset is very small, it’s better to train only the final layers of the network to avoid overfitting, keeping all other layers fixed. So in this case we remove the final layers of the pre-trained network, add new layers and retrain only the new layers. 

When the new dataset is very much large, we can retrain the whole network with initial weights from the pre-trained model. 

***Remark : How to fine tune if the new dataset is very different from the original dataset ?***

The earlier features of a ConvNet contain more generic features (e.g. edge detectors or color blob detectors), but later layers of the ConvNet becomes progressively more specific to the details of the classes contained in the original dataset. The earlier layers can help to extract the features of the new data. So it will be good if we fix the earlier layers and retrain the rest of the layers, if we have only small amount of data. 


### 4.2 Image Feature Extraction 

In this project, transfer learning has been used to extract features from images. The pre-trained model used is the ResNet model which is a model trained on ImageNet dataset [9]. It has the power of classifying upto 1000 classes. ResNet model has skip connections which means the gradients can flow from one layer to another. This means the gradients can also backpropagate easily and hence ResNet model does not suffer from vanishing gradient problem.  Figure 2 shows the architecture of the ResNet model. 

<img src="Image/ResNet Architecture.png" style="width:800px;height:300px;">
