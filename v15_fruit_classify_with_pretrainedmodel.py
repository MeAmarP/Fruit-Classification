'''
Filename: e:/Py Proj/ML/EXPLORES/deep_object_detect/fruits-360/v1_fruit_classify_with_pretrainedmodel.py
Path: e:/Py Proj/ML/EXPLORES/deep_object_detect/fruits-360
Created Date: Tuesday, May 21st 2019, 7:39:33 pm
Author: apotdar

TODO:
    Add data augmentaion and re-train
    Define a function to print classes with probabilty
'''

import os
import tensorflow as tf
from tensorflow.keras import models  
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

base_dir_path = 'E:/Py Proj/ML/EXPLORES/deep_object_detect/fruits-360/'
train_dir_path = os.path.join(base_dir_path,'train')
test_dir_path = os.path.join(base_dir_path,'test')

def getAllClassNames(dir_path):
    return os.listdir(dir_path)

def understandData(BASE_DIR_PATH):    
    train_dir_path = os.path.join(BASE_DIR_PATH,'train')
    #test_dir_path = os.path.join(BASE_DIR_PATH,'test')
    print("Number of Classes = ",len(os.listdir(train_dir_path)))
    AllClassNames = os.listdir(train_dir_path)
    #print("Class Names = ",AllClassNames)
#    print('CLASS NAME'+'\t'+'NUMBER OF IMAGES')    
#    for class_name in AllClassNames:
#        print(class_name+'\t',len(os.listdir(os.path.join(train_dir_path,class_name))))
    displaySampleImages(train_dir_path,AllClassNames)
    return

def displaySampleImages(PATH_TO_TRAIN_DIR,ALL_CLASS_NAMES):
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rcParams['axes.titlesize'] = 8
    import glob
    import cv2
    #NoOfClasses = len(ALL_CLASS_NAMES)   
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.7, wspace=0.1)
    fig.suptitle('Understanding Fruit-360 Dataset', fontsize=16)
    for n,class_name in enumerate(ALL_CLASS_NAMES):
        ImagePath = glob.glob(os.path.join(PATH_TO_TRAIN_DIR,class_name)+'/*.jpg')[0]
        #print(ImagePath)
        Img = cv2.imread(ImagePath)
        ax = fig.add_subplot(10,10,(n+1))
        plt.imshow(cv2.cvtColor(Img, cv2.COLOR_BGR2RGB))
        ax.set_title(class_name+str(n))
        plt.axis('off')
    plt.show()
    return

def readData(BASE_DIR_PATH):
    nb_of_train_files = 0
    nb_of_test_files = 0
    train_dir_path = os.path.join(BASE_DIR_PATH,'train')
    test_dir_path = os.path.join(BASE_DIR_PATH,'test')
    AllClassNames_train = os.listdir(train_dir_path)
    AllClassNames_test = os.listdir(test_dir_path)
    print('Number of Classes in train DataSet: ',len(AllClassNames_train))
    print('Number of Classes in test DataSet: ',len(AllClassNames_test))        
    for class_name in AllClassNames:
        nb_of_train_files = nb_of_train_files + len(os.listdir(os.path.join(train_dir_path,class_name)))
        nb_of_test_files = nb_of_test_files + len(os.listdir(os.path.join(test_dir_path,class_name)))
    print('Number of train samples: ',nb_of_train_files)
    print('Number of test samples:',nb_of_test_files)    
    return

def id_class_name(class_id, classes):
    for key, value in classes.items():
        if class_id == key:
            return value
        
def predictFruitClass(ImagePath,trainedModel,DictOfClasses):
    x = image.load_img(ImagePath, target_size=(150, 150))
    x = image.img_to_array(x)
    #for Display Only
    import matplotlib.pyplot as plt
    plt.imshow((x * 255).astype(np.uint8))
    x = np.expand_dims(x,axis=0)
    x = preprocess_input(x)
    prediction_class = trainedModel.predict_classes(x,batch_size=1)
    prediction_probs = trainedModel.predict_proba(x,batch_size=1)
    class_value = id_class_name(prediction_class,DictOfClasses)
    print(class_value)    
    return prediction_class
    
def getTrainedModel(PATH_TO_TRAINED_MODEL_FILE):
    from tensorflow.keras.models import load_model
    traiendModel = load_model(PATH_TO_TRAINED_MODEL_FILE)
    return traiendModel


AllClassNames = getAllClassNames(train_dir_path)
num_of_classes = len(AllClassNames)
DictOfClasses = {i : AllClassNames[i] for i in range(0, len(AllClassNames))}

conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150,150,3))
conv_base.trainable = False
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dropout(0.5))  
model.add(layers.Dense(num_of_classes,activation='softmax'))

optimizer = Adam(lr=1e-5, decay=1e-6)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
           
model.summary()

datagen = ImageDataGenerator(rescale=1./255,
                             shear_range=0.4,
                             zoom_range=0.2,
                             rotation_range=50,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             horizontal_flip=True,
                             fill_mode='nearest')
batch_size = 48

train_generator = datagen.flow_from_directory(
        train_dir_path,  # this is the target directory
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

test_generator = datagen.flow_from_directory(
        test_dir_path,  # this is the target directory
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels
    
history = model.fit_generator(train_generator,
                          epochs=200,
                          validation_data = test_generator,
                          verbose=1)

model.save('fruit_360_trained_06June.h5')