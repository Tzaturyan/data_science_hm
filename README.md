# data_science_hm
# 1. Calculations using Numpy. Working with data in PandasTask 5**
Create a new column in the authors_price dataframe called cover, it will contain data about which cover this book has - hard or soft. Put the data from the following list in this column:
['hard', 'soft', 'soft', 'hard', 'hard', 'soft', 'soft'].
View the documentation for the pd.pivot_table function using a question mark.For each author, calculate the total cost of hardcover and paperback books. Use the pd.pivot_table function for this. In this case, the columns should be called "hard" and "soft", and the indexes should be the names of the authors. Fill in the missing cost values with zeros, if necessary, download the Numpy library.
Name the resulting dataset book_info and save it in pickle format under the name "book_info.pkl". Then download a dataframe from this file and name it book_info2. Make sure that the book_info and book_info2 dataframes are identical.
Task 1
Import the Numpy library and give it the np alias.
Create a Numpy array called a with a size of 5x2, that is, consisting of 5 rows and 2 columns. The first column must contain numbers 1, 2, 3, 3, 1, and the second one is numbers 6, 8, 11, 10, 7. We will assume that each column is a feature, and the row is an observation. Then find the average value for each attribute using the mean method of the Numpy array. Write the result to the mean_a array, it should have 2 elements.
Task 2
Calculate the a_centered array by subtracting from the values of the array “a” the average values of the corresponding features contained in the mean_a array. The calculation must be performed in one action. The resulting array should have a size of 5x2.
Task 3
Find the scalar product of the columns of the a_centered array. The result should be the value a_centered_sp. Then divide a_centered_sp by N-1, where N is the number of observations.
Task 4**
The number we got at the end of task 3 is the covariance of two features contained in the array “a". In task 4, we divided the sum of the products of centered features by N-1, and not by N, so the value we obtained is an unbiased estimate of covariance.
You can learn more about covariance here:
Selective covariance and selective variance — Studopedia
In this task, check the resulting number by calculating the covariance in another way - using the np.cov function. As an argument to m, the np.cov function must accept the transposed array “a". In the resulting covariance matrix (a 2x2 Numpy array), the desired covariance value will be equal to the element in the row with index 0 and column with index 1.
# Working with data in Pandas
Task 1
Import the Pandas library and give it the alias pd. Create the authors dataframe with the author_id and author_name columns, which respectively contain data: [1, 2, 3] and ['Turgenev', 'Chekhov', 'Ostrovsky'].
Then create a book dataframe with the author_id, book_title and price columns, which respectively contain the data: 
[1, 1, 1, 2, 2, 3, 3],
['Fathers and Children', 'Rudin', 'Noble Nest', 'Fat and Thin', 'Lady with a Dog', 'Thunderstorm', 'Talents and Fans'],
[450, 300, 350, 500, 450, 370, 290].
Task 2
Get the authors_price dataframe by connecting the authors and books dataframes by the author_id field.
Task 3
Create a top5 dataframe that contains lines from authors_price with the five most expensive books.
Task 4
Create an authors_stat dataframe based on the information from authors_price. The authors_stat dataframe should have four columns:
author_name, min_price, max_price and mean_price,
which should contain, respectively, the author's name, minimum, maximum and average price for books by this author.

# 2. Data visualization in Matplotli
Task 1
Download the pyplot module of the matplotlib library with the alias plt, as well as the numpy library with the alias np.
Apply the magic function %matplotlib inline to display graphs in Jupyter Notebook and configure the laptop configuration with the value 'svg' for a clearer display of graphs.
Create a list called x with numbers 1, 2, 3, 4, 5, 6, 7 and a list y with numbers 3.5, 3.8, 4.2, 4.5, 5, 5.5, 7.
Use the plot function to plot a graph connecting points with horizontal coordinates from the x list and vertical coordinates from the y list.
Then, in the next cell, build a scatter plot (other names are scatter plot, scatter plot).
Task 2
Using the linspace function from the Numpy library, create an array t of 51 numbers from 0 to 10 inclusive.
Create a Numpy array called f containing the cosines of the elements of the array t.
Build a line diagram using array t for horizontal coordinates and array f for vertical coordinates. The graph line should be green.
Print the name of the chart - 'Graph f(t)'. Also add names for the horizontal axis - 't Values' and for the vertical - 'f Values'.
Limit the graph on the x-axis to the values 0.5 and 9.5, and on the y-axis to the values -2.5 and 2.5.
*Task 3
Using the linspace function of the Numpy library, create an array x of 51 numbers from -3 to 3 inclusive.
Create arrays y1, y2, y3, y4 using the following formulas:
y1 = x**2
y2 = 2 * x + 0.5
y3 = -3 * x - 1.5
y4 = sin(x)
Using the subplots function of the matplotlib.pyplot module, create a matplotlib.figure.Figure object named fig and an array of Axes objects called ax, and so that you have 4 separate graphs in a grid consisting of of two rows and two columns. In each graph, the array x is used for horizontal coordinates.In the upper-left graph, use y1 for vertical coordinates, y2 in the upper-right, y3 in the lower-left, and y4 in the lower-right.Give the graphs a name: 'Graph y1', 'Graph y2', etc.
For the graph in the upper left corner, set the x-axis boundaries from -5 to 5.
Set the dimensions of the figure to 8 inches horizontally and 6 inches vertically.
The vertical and horizontal gaps between the graphs should be 0.3.
*Task 4
In this task, we will work with a dataset that contains data on credit fraud: Credit Card Fraud Detection (information about the authors: Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi. Calibrating Probability with Undersampling for Unbalanced Classification. In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015).
Read the description and download the creditcard.csv dataset from the website Kaggle.com by the link:
Credit Card Fraud Detection
This dataset is an example of unbalanced data, since fraudulent card transactions are less common than usual.
Import the Pandas library, and also use the “fivethirtyeight” style for graphs.
Use the value_counts method to calculate the number of observations for each value of the target variable Class and apply the plot method to the resulting data to build a bar chart. Then build the same diagram using a logarithmic scale
# Learning with a teacher in Scikit-learn
Task 1
Import the pandas and numpy libraries.
Download the "Boston House Prices dataset" from the built-in datasets of the sklearn library. Create dataframes X and y from this data.
Split these dataframes into training (X_train, y_train) and test (X_test, y_test) using the train_test_split function so that the size of the test sample
is 30% of all data, while the random_state argument should be equal to 42.
Create a linear regression model called lr using the LinearRegression class from the sklearn.linear_model module.
Train the model on the training data (use all the features) and make a prediction on the test data.
Calculate the R2 of the received predictions using r2_score from the sklearn.metrics module.
ask 2
Create a model called model using RandomForestRegressor from the sklearn.ensemble module.
Make the n_estimators parameter equal to 1000,
max_depth should be equal to 12 and make random_state equal to 42.
Train the model on the training data in the same way as you trained the LinearRegression model,
but at the same time put y_train in the fit method instead of the y_train dataframe.values[:, 0]
to get a one-dimensional Numpy array from the dataframe,
since for the RandomForestRegressor class in this method, it is preferable to use arrays instead of a dataframe for the y argument.
Make a prediction on the test data and calculate R2. Compare with the result from the previous task.
Write in the comments to the code which model works better in this case.
*Task 3
Call the documentation for the RandomForestRegressor class,
find information about the feature_importances_ attribute.
Use this attribute to find the sum of all the indicators of importance,
set which two signs show the greatest importance.
*Task 4
In this task, we will work with a dataset that we are already familiar with from the Matplotlib library homework, this is the Credit Card Fraud Detection dataset.For this dataset, we will solve the classification problem - we will determine which of the credit card transactions are fraudulent.This dataset is highly unbalanced (since fraud cases are relatively rare), so using the accuracy metric will not be useful and will not help you choose the best model.We will calculate AUC, that is, the area under the ROC curve.
Import from the corresponding modules RandomForestClassifier, GridSearchCV and train_test_split.
Upload the creditcard.csv dataset and create a df dataframe.
Using the value_counts method with the normalize argument=True make sure that the selection is unbalanced. Using the info method, check whether all columns contain numeric data and whether there are any gaps in them.Apply the following setting so that you can view all the columns of the dataframe:
pd.options.display.max_columns = 100.
View the first 10 lines of the df dataframe.
Create dataframe X from dataframe df by excluding the Class column.
Create a Series object called y from the Class column.
Split X and y into training and test datasets using the train_test_split function, using the arguments: test_size=0.3, random_state=100, stratify=y.
You should get the objects X_train, X_test, y_train and y_test.
View the information about their form.
To search through the parameters grid, set the following parameters:
parameters = [{'n_estimators': [10, 15],
'max_features': np.arange(3, 5),
'max_depth': np.arange(4, 7)}]
Create a GridSearchCV model with the following arguments:
estimator=RandomForestClassifier(random_state=100),
param_grid=parameters,
scoring='roc_auc',
cv=3.
Train the model on a training dataset (may take a few minutes).
View the parameters of the best model using the best_params_ attribute.
Predict the probabilities of classes using the resulting model and the predict_proba method.
From the resulting result (Numpy array), select a column with index 1 (probability of class 1) and write y_pred_proba to the array. Import the roc_auc_score metric from the sklearn.metrics module.
Calculate the AUC on the test data and compare it with the result obtained on the training data, using the arrays y_test and y_pred_proba as arguments.
*Additional tasks:
1). Load the Wine dataset from the built-in sklearn.datasets datasets using the load_wine function into the data variable.
2). The resulting dataset is not a dataframe. This is a data structure that has keys similar to a dictionary. View the data type of this data structure and create a list of data_keys containing its keys.
3). View the data, description and names of features in the dataset. The description should be output in the form of a familiar, neatly designed text, without line break symbols, but with the hyphenations themselves, etc. 4). How many classes does the dataset target variable contain? 
You will see the names of the classes.
5). Based on the dataset data (they are contained in a two-dimensional Numpy array) and feature names, create a dataframe called X.
6). Find out the size of the dataframe X and determine if there are missing values in it.
7). Add to the dataframe a field with wine classes in the form of numbers having the data type numpy.int64. The field name is 'target'.
8). Construct a correlation matrix for all fields of X. Give the resulting dataframe the name X_corr.
9). Create a list of high_corr features whose correlation with the target field by absolute value exceeds 0.5 (moreover, the target field itself should not be included in this list).
10). Remove the target variable field from the dataframe X. For all the features whose names are contained in the high_corr list, calculate the square of their values and add to the dataframe X the corresponding fields with the suffix '_2' added to the original name of the feature. The final dataframe should contain all the fields that were originally in it, as well as fields with attributes from the high_cor list
# Learning without a teacher in Scikit-learn.
Task 1
Import the pandas, numpy, and matplotlib libraries.
Download the "Boston House Prices dataset" from the built
-in datasets of the sklearn library.
Create dataframes X and y from this data.
Split these dataframes into training (X_train, y_train) and test (X_test, y_test)
using the train_test_split function so that the size of the test sample
is 20% of all data, while the random_state argument should be equal to 42.
Scale your data using StandardScaler.
Build a TSNE model on training data with parameters:
n_components=2, learning_rate=250, random_state=42.
Build a scatter plot on this data.
Task 2
Using KMeans, split the data from the training set into 3 clusters,
use all the features from the X_train dataframe.
The max_iter parameter should be equal to 100, make random_state equal to 42.
Plot the scatter plot again on the data obtained using TSNE,
and color the points from different clusters with different colors.
Calculate the average values of price and CRIM in different clusters.
*Task 3
Apply the KMeans model built in the previous task
to the data from the test set.
Calculate the average values of price and CRIM in different clusters on the test data.
#Consultation on the final project
Delivery of the course project:
Attach a link to the solution (if the link to the laptop is on kaggle, make it public)
Specify the nickname on kaggle, which is displayed on the leaderboard.
