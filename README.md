# Fruit-Classification
Fruit Classification using TensorFlow-Keras on Fruits 360 dataset

## Understand Dataset:
![Understanding Dataset][EDA_Img]

[EDA_Img]: https://github.com/MeAmarP/Fruit-Classification/blob/master/results/EDA_images_v22.png
 
## Approch
+ I used MobileNetV2 architecutre, pre-trained on ImageNet dataset for feature extraction.

+ Next I use these features and ran through a new classifier, which is trained from scratch.

+ As stated in my Favourite Book: __Deep Learning with Python__. 
We took convolitional base(conv_base) of Mobilenet, ran new data through it and trained a new classifier on top of
the output.

So basically, I extended the conv_base by adding Dense layer followed by DropOut layer, and running 
whole network on input data with data augmentation. well this is bit expensive, but meh!! I have enough processing power.

+ Important Thing, I freeze the convolutional base so as to avoid updating their weights.


TODO:
+ Add Image of EDA and assocoated written functions.
+ Add docStrings for each of the function you have written


## Refrences:
+ MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
 <https://arxiv.org/abs/1704.04861>
+ Deep Learning with Python, François Chollet.