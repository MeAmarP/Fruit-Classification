# Fruit-Classification: Work-In-Progress
Fruit Classification using TensorFlow-Keras on Fruits 360 dataset

## Understand Dataset:
![Understanding Dataset][EDA_Img]

[EDA_Img]: https://github.com/MeAmarP/Fruit-Classification/blob/master/results/EDA_images_v22.png

### Step 1 - EDA:

__Method:__
```python
readData(base_dir_path)
```
__Console Output:__
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
__Console Output:__
```console
CLASS NAME          NUMBER OF IMAGES
Apple Braeburn      492
Apple Golden 1      492
Apple Golden 2      492
Apple Golden 3      481
Apple Granny Smith  492
```

## Build Model and Train Dataset:

### Approch:
+ I used MobileNetV2 architecutre, pre-trained on ImageNet dataset for feature extraction.
+ Next I use these features and ran through a new classifier, which is trained from scratch.
+ As stated in my Favourite Book: __Deep Learning with Python__. 
We took convolitional base(conv_base) of Mobilenet, ran new data through it and trained a new classifier on top of
the output.
+ So basically, I extended the conv_base by adding Dense layer followed by DropOut layer, and running 
whole network on input data with data augmentation. 
+ Well!! this is computationally bit expensive, but meh!! I have enough processing power.
+ Important Thing, I freeze the convolutional base so as to avoid updating their weights.

### Step 2 - Compiling Model:
__Method:__
```python
compileClassifyModel(num_of_classes)
```
__Console Output:__
```console
Layer (type)                 Output Shape              Param #   
=================================================================
mobilenetv2_1.00_224 (Model) (None, 1280)              2257984   
_________________________________________________________________
flatten_1 (Flatten)          (None, 1280)              0         
_________________________________________________________________
dense_2 (Dense)              (None, 512)               655872    
_________________________________________________________________
dropout_1 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 95)                48735     
=================================================================
Total params: 2,962,591
Trainable params: 704,607
Non-trainable params: 2,257,984
_________________________________________________________________
```

### Step 3 - Training compiled Model:

### Training Results:
**Epcohs:20**
![train_valid_acc][plot_acc]

[plot_acc]: https://github.com/MeAmarP/Fruit-Classification/blob/master/results/train_valid_acc_16JUL_20epochs.png

![train_valid_loss][plot_loss]

[plot_loss]: https://github.com/MeAmarP/Fruit-Classification/blob/master/results/train_valid_Loss_16JUL_20epochs.png


### TODO:
+ Test with more epochs.
+ add method to Evaluate prediction accuracy and loss on whole test dataset


## Refrences:
+ MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
 <https://arxiv.org/abs/1704.04861>
+ Deep Learning with Python, François Chollet.