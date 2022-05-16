## 1. Assignment 4 - self-assignment: emotion classification 
Discription of what the script does and what kind of problem it tries to solve 

## 2. Methods

Download data from command line: ```wget -O in/fer2013.csv https://www.dropbox.com/s/zi48lkarsg4kbry/fer2013.csv\?dl\=1```
or from this link: https://www.dropbox.com/s/zi48lkarsg4kbry/fer2013.csv\?dl\=1

The ```load_dataset()``` function was found through https://colab.research.google.com/github/RodolfoFerro/PyConCo20/blob/full-code/notebooks/Deep%20Learning%20Model.ipynb#scrollTo=9v3fuYQb139s and further adapted. 

The ```mdl()``` function was found through: https://www.kaggle.com/code/aayushmishra1512/emotion-detector. and was further adapted. I tested different parameters and layers and found that the code found through this link, and reducing the epochs to 20, produced the highest validation data accuracy as well as the tightest fit between both the loss function and accuracy for the test and validation data. However, the code allows the user to define parameters such as epochs and batch size. 

## 3. Usage ```emotion_classification.py```
To run the code:
- Pull this repository with this folder structure
- Install the packages mentioned in ```requirements.txt```
- Set your current working directory to the level above ```src```
- Download the data and place it in ```in```
- Write in the command line: ```python src/emotion_classification.py + argparse parameters```

## 4. Discussion of Results 
The model has an overall accuracy of 60%, though reaching 63% on other runs, on the validation data. However, there are large differences on how the model performs on the individual classes. It classifies fear at 47% and happy at 83%. Furthermore, the model classifies happy and sad best by far, which suggests that the model would perhaps very accurately on a binary classification task between those two classes. 

I tested different parameters and layers, where especially adding more epochs made the loss functions and accuracies of the validation and testing data to diverge. The code that produced the plots and classification report in ```out``` return the highest accuracy of all as well as the best fit between loss and accuracy.

The images are grayscale which decreases the complexity the model can have. Furthermore, the images are very pixilated which could also impact the model's performance.
