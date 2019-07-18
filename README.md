# Fruit-Classification: Work-In-Progress
Fruit Classification using TensorFlow-Keras on Fruits 360 dataset

## Understand Dataset:
![Understanding Dataset][EDA_Img]

[EDA_Img]: https://github.com/MeAmarP/Fruit-Classification/blob/master/results/EDA_images_v22.png

### EDA:

__Method:__
```python
readData(base_dir_path)
```
__Output:__
```console
Total Number of Classes in train DataSet:  95
Total Number of Classes in test DataSet:  95
Total Number of train samples:  48905
Total Number of test samples: 16421
```
__Method:__
```python
understandData(base_dir_path,'train')
```
__Output:__
```console
CLASS NAME          NUMBER OF IMAGES
Apple Braeburn      492
Apple Golden 1      492
Apple Golden 2      492
Apple Golden 3      481
Apple Granny Smith  492
```

## Approch:
+ I used MobileNetV2 architecutre, pre-trained on ImageNet dataset for feature extraction.

+ Next I use these features and ran through a new classifier, which is trained from scratch.

+ As stated in my Favourite Book: __Deep Learning with Python__. 
We took convolitional base(conv_base) of Mobilenet, ran new data through it and trained a new classifier on top of
the output.

So basically, I extended the conv_base by adding Dense layer followed by DropOut layer, and running 
whole network on input data with data augmentation. well this is bit expensive, but meh!! I have enough processing power.

+ Important Thing, I freeze the convolutional base so as to avoid updating their weights.


## Results:
Epcohs:20

![train_valid_acc][plot_acc]

[plot_acc]: https://github.com/MeAmarP/Fruit-Classification/blob/master/results/train_valid_acc_16JUL_20epochs.png

![train_valid_loss][plot_loss]

[plot_loss]: https://github.com/MeAmarP/Fruit-Classification/blob/master/results/train_valid_Loss_16JUL_20epochs.png


### TODO:
+ Test with more epochs.
+ add method to Evaluate prediction accuracy and loss on whole test dataset
+ Add docStrings for each of the function you have written.



## Refrences:
+ MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
 <https://arxiv.org/abs/1704.04861>
+ Deep Learning with Python, François Chollet.