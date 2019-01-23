# ImageCategorization
## IC Task Introduction
For the IC task, you need to design and implement an automatic classification system that can predict the label of each input image. In this task, each image can be classified into one of 20 categories. The predefined categories describe the main object of the images. These categories include bird, boat, person, tv monitor etc
### Training Data
1. The training data includes about 32,000 images and their labels. The label of each image can be found in file train.label (in directory: ic-data\train). For convenience, we just use different numbers (from 1 to 20) to represent different categories in this task. Each row of this file is of the form <image ID #label# >. Elements in a row are separated by whitespace.
2. There might be some noises in the training data. You may need to identify and repair these noises.
### Evaluation Data
The evaluation file (test data) will not be provided until laboratory session. The test data will only include some unlabeled images without noise, and there will be no extra file that provides the label of each image.
