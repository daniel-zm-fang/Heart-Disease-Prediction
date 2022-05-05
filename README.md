# Heart-Disease-Prediction

This is the [link](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease) to the dataset used.

## Goal
To train a ML classifier on the above dataset to predict whether a patient has heart disease or not. The classifer should have a high recall (also preferably high precision) because we want to predict all the patients with heart disease to be "Yes" (it's better to have more false positives than false negatives in this case because heart diease is a serious matter).

## Setup
```bash
pip install -r requirements.txt
```

## Structure
The progression of notebooks is as follows:
1. exploration_1.ipynb: Exploring the dataset
2. exploration_2.ipynb: Balancing the dataset using imblearn
3. exploration_3.ipynb: Continue to balance the dataset and confirm plot precison-recall curve
4. exploration_4.ipynb: Try balancing with sklearn models' "class_weight" parameter as alternative to imblearn; Fine-tuning XGBClassifier and LogisticRegression
5. voting_clf is the final model

## Results
The final model (saved as voting_clf.pkl) has a precision of 0.31, recall of 0.55, and F1 score of 0.39.
<br>
For comparison:
* the logistic regression model, without balancing class weights, has precison of 0.55, recall of 0.11, and F1 score of 0.18
* the fine-tuned XGBClassifier has a precision of 0.22, recall of 0.77, and F1 score of 0.35

## Commments
* More diverse models can be added to the voting_clf to further improve the F1 score. I tried fine tuning SVC (with linear kernel) but the tuning continues even after a day, so I did not include SVC in the voting_clf.
* The majority of this project is focused on increasing the recall of the model while trying to minimize the sacrifice of precision. If recall is not the goal, then there is no need to using imblearn as some models already have a decent precision (around 0.55).
* I thought about using stacking. But the mosting promising model (XGBClassifier) is already a complex base model by itself. Thus I don't think stacking will bring much, if any improvement.
* I don't think some of the features are very useful. For example, features like MentalHealth (Thinking about your mental health, for how many days during the past 30 days was your mental health not good?) is subjective. Everyone rate their mental state differently.
* To improve the model, more postive samples and more objective features (ex. BMI) will definitely help with the models.
