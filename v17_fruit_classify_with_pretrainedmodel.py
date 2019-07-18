'''
Created Date: Tuesday, May 21st 2019, 7:39:33 pm
Author: MeAmarP

Description: 
    Fruit Classification using Pre-trained convnet for feature extraction
    
Dataset: <https://www.kaggle.com/moltean/fruits>
'''

import os
import tensorflow as tf
from tensorflow.keras import models  
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

from datetime import date

base_dir_path = os.getcwd()
train_dir_path = os.path.join(base_dir_path,'train')
test_dir_path = os.path.join(base_dir_path,'test')
MNet_InputSize = (224,224)

def getAllClassNames(dir_path):
    """
        Returns list of all class names in given train/test dir path.
    """
    return os.listdir(dir_path)

def understandData(BASE_DIR_PATH,train_or_test):
    """
    Function prints number of images per class in train/test directory
    <CLASS-NAME    NUMBER-OF-IMAGES>
    
    Args:
        BASE_DIR_PATH(str): path of the base directory
        train_or_test(str): directory to select train/test
    """    
    train_dir_path = os.path.join(BASE_DIR_PATH,train_or_test)
    #test_dir_path = os.path.join(BASE_DIR_PATH,'test')
    print("Number of Classes = ",len(os.listdir(train_dir_path)))
    AllClassNames = os.listdir(train_dir_path)
    #print("Class Names = ",AllClassNames)
    print('CLASS NAME'+'\t'+'NUMBER OF IMAGES')    
    for class_name in AllClassNames:
        print(class_name+'\t',len(os.listdir(os.path.join(train_dir_path,class_name))))
    print("======================================================================")
    #displaySampleImages(train_dir_path,AllClassNames)
    return

def readData(BASE_DIR_PATH):
    """
    Console output of,
        total number of classes in train/test dir 
        total number of images in train/test dir
    in given dataset.
    
    Args:
        BASE_DIR_PATH(str): path to root dir
    """
    nb_of_train_files = 0
    nb_of_test_files = 0
    train_dir_path = os.path.join(BASE_DIR_PATH,'train')
    test_dir_path = os.path.join(BASE_DIR_PATH,'test')
    AllClassNames_train = os.listdir(train_dir_path)
    AllClassNames_test = os.listdir(test_dir_path)
    print('Total Number of Classes in train DataSet: ',len(AllClassNames_train))
    print('Total Number of Classes in test DataSet: ',len(AllClassNames_test))        
    for class_name in AllClassNames:
        nb_of_train_files = nb_of_train_files + len(os.listdir(os.path.join(train_dir_path,class_name)))
        nb_of_test_files = nb_of_test_files + len(os.listdir(os.path.join(test_dir_path,class_name)))
    print('Total Number of train samples: ',nb_of_train_files)
    print('Total Number of test samples:',nb_of_test_files)    
    return

def displaySampleImages(PATH_TO_DIR,ALL_CLASS_NAMES):
    """
    Display grid of sample images for every class in dataset.
    
    Args:
        PATH_TO_DIR(str): path to train or test dir.
        ALL_CLASS_NAMES(str): list of all class names.
        
    """
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
        ImagePath = glob.glob(os.path.join(PATH_TO_DIR,class_name)+'/*.jpg')[0]
        #print(ImagePath)
        Img = cv2.imread(ImagePath)
        ax = fig.add_subplot(10,10,(n+1))
        plt.imshow(cv2.cvtColor(Img, cv2.COLOR_BGR2RGB))
        ax.set_title(class_name)
        plt.axis('off')
    plt.show()
    return

def compileClassifyModel():
    conv_base = MobileNetV2(weights='imagenet',
                        include_top=False,
                        pooling='avg')
    conv_base.trainable = False #Freeze Model 
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
    return model

def trainClassifyModel(ClassifyModel,
                       batch_size=16,
                       save_model_filename='fruit_classify_model',
                       MNet_InputSize=(224,224)):
    
    datagen = ImageDataGenerator(rescale=1./255,
                             shear_range=0.4,
                             zoom_range=0.2,
                             rotation_range=50,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             horizontal_flip=True,
                             fill_mode='nearest')
        
    train_generator = datagen.flow_from_directory(
            train_dir_path,  # this is the target directory
            target_size=MNet_InputSize,  # all images will be resized
            batch_size=batch_size,
            class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels
    
    test_generator = datagen.flow_from_directory(
            test_dir_path,  # this is the target directory
            target_size=MNet_InputSize,  # all images will be resized
            batch_size=batch_size,
            class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels
        
    history = ClassifyModel.fit_generator(train_generator,
                              epochs=20,
                              validation_data = test_generator,
                              verbose=1)
    trainedModel_Filename = SaveModelFile(ClassifyModel,save_model_filename)   
    return history,trainedModel_Filename

def plotTrainResults(history):
    """
    Summary:
        Plots train-validation loss and accuracy graphs for given history object
    
    Args:
        historyObject(object): Object returned by fit/fit_generator during train.        
    """
    import matplotlib.pyplot as plt
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
    return

def SaveModelFile(ClassifyModel,save_model_filename):
    today = date.today()
    date_str = today.strftime("%d%m%y")
    save_model_filename = '_'.join([save_model_filename,date_str])
    save_model_filename = save_model_filename +'.h5'
    ClassifyModel.save(save_model_filename)
    print('Done Saving Model File...')
    return save_model_filename

def getTrainedModel(PATH_TO_TRAINED_MODEL_FILE):
    """
    Loads trained-saved model from file(.h5) and returns as a object.
    
    Args:
        PATH_TO_TRAINED_MODEL_FILE(str): path to saved model file.
    
    returns:
        trainedModel(model object): returns a model saved as a <.h5> 
    """
    from tensorflow.keras.models import load_model
    traiendModel = load_model(PATH_TO_TRAINED_MODEL_FILE)
    return traiendModel

def id_class_name(class_id, classes):
    """
    Returns name of the class as per the given id
    
    Args:
        class_id(int): Number of the class.
        classes(dict): dictinary of all the classes in given dataset.
        
    returns:
        Name of the Class.
    """
    for key, value in classes.items():
        if class_id == key:
            return value
        
def predictFruitClass(ImagePath,trainedModel,DictOfClasses):
    """
    Perform class prediction on input image and print predicted class.
        
    Args:
        ImagePath(str): Absolute Path to test image
        trainedModel(object): trained model from method getTrainedModel()
        DictOfClasses(dict): python dict of all image classes.
        
    Returns:
        Probability of predictions for each class.        
    """
    x = image.load_img(ImagePath, target_size=MNet_InputSize)
    x = image.img_to_array(x)
    #for Display Only
    import matplotlib.pyplot as plt
    plt.imshow((x * 255).astype(np.uint8))
    x = np.expand_dims(x,axis=0)
    x = preprocess_input(x)
    prediction_class = trainedModel.predict_classes(x,batch_size=1)
    prediction_probs = trainedModel.predict_proba(x,batch_size=1)
    #print(prediction_probs)
    class_value = id_class_name(prediction_class,DictOfClasses)
    print(class_value)
    return prediction_probs
    

#==============================================================================


if __name__ == "__main__":    
    base_dir_path = os.getcwd()
    train_dir_path = os.path.join(base_dir_path,'train')
    test_dir_path = os.path.join(base_dir_path,'test')
    MNet_InputSize = (224,224)
        
    AllClassNames = getAllClassNames(train_dir_path)
    num_of_classes = len(AllClassNames)
    DictOfClasses = {i : AllClassNames[i] for i in range(0, len(AllClassNames))}
    
    readData(base_dir_path)
    understandData(base_dir_path,'train')
    displaySampleImages(train_dir_path,AllClassNames)
    classifyModel = compileClassifyModel()
    trainingHistory,trainedModel_filename = trainClassifyModel(classifyModel)
    plotTrainResults(trainingHistory)
    
    # Model Prediction on test Images.
    ImagePath = 'test/Banana Red/99_100.jpg'
    path_trained_model = os.path.abspath(trainedModel_filename)
    trainedModel = getTrainedModel(path_trained_model)
    AllProbs = predictFruitClass(ImagePath,trainedModel,DictOfClasses)