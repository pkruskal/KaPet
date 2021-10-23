# KaPet

## submissions and output
Models are submitted as jupyter notebooks so all submissions should be in that form and saved in `/submissions`. 
Packages and code can be uploaded to Kaggles servers and then imported in your notebook.
Idealy the notebook is mostly just calling your classes and functions

## Generic configuration parameters
- `image_dimention`: 128 # dimention to force the image into
- `device`: "cpu"  # can be either (cpu / cuda / cuda:) depending on your system. cuda followed by a number specifies the gpu to run on
- `rotation_augmentations`: 50  # +/- in degrees # for rotating the image for data augmentation
- `translation_augmentations`: 0  # % translation of image before resizing # for translateing the image for data augmentation
  
# Models

## simple NN backbone followed by fully connected network predictor
- build a simple CNN (or image transformer) to generate a feature array. 
- add on features of the metadata and 


### specific configurations
- `model_configuraton` ("VGG19_512_32_1") Specifys the model to use for the NN backbone
- `batchsize`: 10 # how many images to use per learning rule update
- `epocs` : 10 # how many runs through the data to train on
- `learning_rate` : 0.001 # a single number will use the Adam optimizer, 
  can also be a schedule of trainign of epocs and learning rates for using SGD
- `learning_rate` : {
      "epocs" : [3,6,9],
      "rates" : [0.001,0.0003,0.00001]
  }
  
