# Logistic Regression with Titanic Dataset


The aim of this notebook is to use every trick that I know about Logistic Regression, from data preprocessing to feature selection and so on, to achieve performance that is comparable to other more powerful algorithms such as Random Forest. Through this I want to prove that there are a lot more to analysing a dataset than just using the most powerful algorithms out there.

The dataset in question is the famous [Titanic dataset](https://www.kaggle.com/c/titanic/overview). We are going 
## 1. Load the dataset and some basic preprocessing

![screen](pic/df_1.png)

The raw data of the first 10 rows is presented above. Immediately there are a few things we can do:
  - Remove the columns that is not necessary for our analysis: PassengerId, Name and Ticket. 
  - Fill out the NULLs in each column or remove the column entirely if it is too sparse.

Removing PassengerId is a no-brainer, since it is just the index of each row and convey . However,
