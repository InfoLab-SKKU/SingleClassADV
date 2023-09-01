# SingleADV: Single-class Target-specific Attack Against Interpretable Deep Learning Systems

Deep learning techniques have achieved state-of-the-art performance in different domains. It is always essential to verify that high confidence in a specific task is the result of correct modeling of the addressed problems. In this case, interpretation models play a crucial role in developing deep learning models. Interpretation models help understand the inner working of DNN models. However, interpretations of deep learning models are susceptible to adversarial manipulations. We propose SingleADV, a single-class target-specific attack that generates a universal perturbation to fool the target DNN model to confuse a whole category of objects (i.e., specific category) with a target category in both white-box and black-box scenarios. SingleADV, simultaneously, misleads the interpretation model by enabling adversarial attribution maps similar to their corresponding benign. The attack also limits unintended fooling by samples from other categories.

## Software Requirements
* Foolbox 1.8.0 (https://github.com/bethgelab/foolbox)

The code is based on the CAM interpreter. The attack uses ResNet50 by default. 

## Data Downloading
Download the ImageNet ILSVRC2012 dataset from http://image-net.org/download (training and validation dataset). One needs to 
register before downloading. To set up the data for our code, we will create directories named by the winds, and inside each directory, there will be two folders for training and testing. The training images will be inside the training folder, and the
validation images will become part of the testing folder.

The testing images will come from the imageNet validation dataset, which provides 50 samples for each class. The training images will
come from the imageNet training data. This dataset can be created by picking 50 samples from each class (wnid). Once the data is in this shape, you can open the code folder and find the 
file __config.ini__; this file has to be modified to include the relevant paths.

Below, there is a description of each field present in the *config.ini*. 

| S/N | Field         | Field Description  |
| ----|-------------| ------------------|
|1    | saveDir       | Path of the directory where the results will be saved. The results are saved as “AdversarialAttackResults.db” |
|2    | datasetDir    | Path of the directory where the dataset will be present, folders named by wnids and inside each folder we should have testing and training folder.  |
|3    | imageNetValidationDir    | Path of the directory where imagenet validation images can be found. There are 50000 images. |
|4    | imageNet2012ValidationGroundTruthFile | Path of the file “ILSVRC2012_validation_ground_truth.txt”. This comes with ImageNet2012 validation dataset. |
|5    | imageNet2012LabelMapFile  | Path of the file “imagenet_2012_challenge_label_map_proto.pbtxt”. This comes with the imageNet2012 validation dataset. |
|6    | sourceIdentities   | It is a comma separated Wnids that will be taken as source classes. Note the data will be picked based on these wnids and the path of the dataset set in datasetDir. |
|7    | targetIdentities   | It is comma separated Wnids that will be taken as target classes.|
|8    | attackModels       | Comma separated attack Model Ids. It represents the deep model for launching the target attack. You can find the table below to select it. |
|9   | etas               | Comma separated values of eta for each algorithm id.|
|10   | algorithmId        | Comma separated Algorithm IDs. These algorithms will be launched one by one on each deep Models that you have select on each pair of source and target Identities. Please see the table below to find the algorithm ids. |

The algorithmIds can be selected from table below.

| S/N | Algorithm Description   | Algorithm ID |
| ----|-------------------------| -------------|
|1    | LinfinityBounded        | 3            |
|2    | L2Bounded               | 4            |

and the attack models can be selected from the below table

| S/N | AttackModel Description | Attack Model ID |
| ----|-------------------------| ----------------|
|1    | VGG16                   | 1               |
|2    | ResNet50                | 2               |
|3    | InceptionV3             | 3               |
|4    | MobileNetV2             | 4               |


Once you have setup the config.ini file, you run the code by running the script as 
 ```
 python attack.py 
 
 ```
 
The results are saved in the database. One can check the tables *attacktrainingperformance* and *attacktestingperformance* 
to find the training and testing accuracy. The perturbations are saved in *attack* table, in the column *perturbedimage*.

![alt text](https://github.com/EldorToptal/SingleClassADV/blob/main/SingleClassADV/attack_main_idea_example-1.png?raw=true)
