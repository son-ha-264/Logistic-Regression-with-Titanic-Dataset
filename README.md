# Logistic Regression with Titanic Dataset using R

The dataset in question in question is the famous [Titanic dataset](https://www.kaggle.com/c/titanic/overview). We are going to perform some data preprocessing/cleaning and fit a Logistic Regression model to predict whether a passenger survivied the accident or not. Information on the dataset and the meaning of each variables can be found in the hyperlink.

## 1. Load the dataset and some basic preprocessing

![screen](pic/df_1.png)

Combining the train set and test set, the raw data of the first 10 rows is presented above. Immediately there are a few things we can do:
  - Remove the columns that is not necessary for our analysis: **PassengerId**, **Name** and **Ticket**. 
  - Deal with missing data. We will discuss this in details in the section below.
  
### 1.1. Dealing with missing data

There are 4 columns with missing data: **Age**,**Fare**,**Cabin**,**Embarked**. **Cabin** has 1014 missing values in a total of 1309 data points. We remove this column because it is so sparse. **Embarked** is a categorical variable with 2 missing value, hence we impute with the most common value 'S'. **Fare**, on the other hand, is a numerical variable with one missing value, which we impute using the mean of the entire column to preserve the mean after imputation.

**Age** is a numerical value as well. But if we apply mean imputation we can see the distribution of Age changes drastically. This is because of higher number of missing values compared to **Fare**.

![screen](pic/hist_1.png)

Let's try another imputation method: Multivariate Imputation by Chained Equation ([MICE](https://cran.r-project.org/web/packages/mice/mice.pdf)). MICE assumes that the data is Missing At Random, meaning the probability of missing data of a variable is dependent on the other variables. Hence, say if we have variables X1,X2,...Xn, and X1 is the missing value, we can predict what X1 is using a model trained on X2,..,Xn. 

Imputing **Age** with MICE results in almost the same distribution of Age, which is a much better result.

![screen](pic/hist_2.png)

## 2.Exploratory Analysis

<p float="left">
  <img src="pic/plot_sex.jpg" width="250" />
  <img src="pic/plot_pclass.jpg" width="250" /> 
  <img src="pic/plot_sibsp.jpg" width="250" /> 
</p>
<p float="left">
  <img src="pic/parch.png" width="400" />
  <img src="pic/plot_embarked.png" width="400" /> 
</p>
<p float="left">
  <img src="pic/plot_age.png" width="400" />
  <img src="pic/plot_fare.png" width="400" /> 
</p>

The first 5 bar charts are counts of survivors and non-survivors with respect to **Sex**, **Pclass**,**SibSp**,**Parch** and **Embarked**. The last two are scatterplots of Survival against Age and Fare. We can observe that women, higher class ticket owners and those embarked from Cherbourg and Southhampton were more likely to survive. Ticket fare does not seem not say much about survival probability. Lastly, those with only one Parent/Children or Sibling/Spouse were less likely to survive. We create another variable **SibSp_1** and **Parch_1** equals to 1 if **SibSp**,**Parch** = 1 and 0 otherwise. 

We can try to feature engineer another variable 'Family Size' ( idea borrowed from this [this fantastic notebook](https://www.kaggle.com/mrisdal/exploring-survival-on-the-titanic/)), which is calculated by: **FullSize** = 1 + **SibSp** + **Parch**. Plotting the bar chart reveals an interesting pattern: family with size 2, 3, 4 were more likely to survive than other family size (either singleton or more than 4). To better reflect this observation, we set family size variable **Fsize** = 'medium' if family has size 2, 3, 4, 'single' if family group only has one member, and 'large' if family has size more than 4.

![screen](pic/plot_fsize.png)

We end up with the below table ready for model fitting and prediction.

![screen](pic/df_3.png)

## 3. Model Fitting and Variable Selection

Machine Learning approach aims to find the parameters to minimise an Objective function. On the other hand the statistical approach finds the Maximum Likelihood Estimator (MLE) of the parameters. Mathematically when it comes to Logistic Regression, both approaches are very much the same, since minimising the empirical risk associated with the log loss (ML objective) is equivalent to maximising the Likelihood function (Statistical method objective). For this notebook I will perform model selection using statistical methods in R.

At first fitting Logistic Regression using all variables as explanatory variables to predict **Survived**, we obtain 2 tables below, one is the summary table, and the other is the Analysis of Deviance table: 

