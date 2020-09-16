# Automatic Image Captioning Using CNNs

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

Generating captions for images is a vital task relevant to the area of both Computer Vision and Natural Language Processing. Mimicking the human ability of providing descriptions for images by a machine is itself a remarkable step along the line of Artificial Intelligence. The main challenge of this task is to capture how objects relate to each other in the image and to express them in a natural language (like English).Traditionally, computer systems have been using pre-defined templates for generating text descriptions for images. However, this approach does not provide sufficient variety required for generating lexically rich text descriptions. This shortcoming has been suppressed with the increased efficiency of neural networks. Many state of art models use neural networks for generating captions by taking image as input and predicting next lexical unit in the output sentence. Some real world scenarios where Image Captioning plays a vital role are as follows :
* 1. Self driving cars — Automatic driving is one of the biggest challenges and if we can properly caption the scene around the car, it can give a boost to the self driving system.
* 2. Aid to the blind — We can create a product for the blind which will guide them travelling on the roads without the support of anyone else. We can do this by first converting the scene into text and then the text to voice. Both are now famous applications of Deep Learning.
* 3. CCTV cameras are everywhere today, but along with viewing the world, if we can also generate relevant captions, then we can raise alarms as soon as there is some malicious activity going on somewhere. This could probably help reduce some crime and/or accidents.
* 4. Automatic Captioning can help, make Google Image Search as good as Google Search, as then every image could be first converted into a caption and then search can be performed based on the caption.


