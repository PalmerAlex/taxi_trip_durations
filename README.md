## 1.0 Abstract

In this study, we address the problem of predicting taxi trip durations using a dataset comprising of various features such as geocoordinates
and dates. We implement and compare two different models: a deep learning model using an Artificial Neural Network (ANN) and a traditional
machine learning model using Random Forest Regression. The dataset undergoes extensive pre-processing, including feature extraction,
dimensionality reduction, and removal of outliers, ultimately leading to high accuracy in predictions, as evidenced by the impressive R2 score.
Our findings indicate that the combination of advanced data processing techniques and the appropriate choice of modelling methods can
significantly improve taxi trip duration predictions, with potential applications in transportation planning and optimization.

## 2.0 Background

Predicting taxi trip durations is an essential task in the transportation industry, as it streamlines the optimization of taxi services, reduces
passenger waiting times, and enhances overall transportation efficiency. Numerous factors, including traffic conditions, weather, and
geographic features, contribute to the challenge of accurately predicting taxi trip durations.

Existing research on taxi trip duration prediction has utilized a variety of machine learning techniques, ranging from traditional approaches
such as linear regression and decision trees to more advanced methods like deep learning [1]. However, there is no agreement on the most
effective strategy. The role of data pre-processing and feature engineering in improving prediction accuracy remains an active area of
investigation [2].

This study aims to augment the current body of knowledge by comparing the performance of an Artificial Neural Network (ANN)-based deep
learning model and a Random Forest Regression model in predicting taxi trip durations. Moreover, we examine the impact of advanced data
pre-processing techniques, such as dimensionality reduction and time data transformation, on the performance of these models.

In conclusion, our work seeks not only to determine the most effective modelling technique for predicting taxi trip durations but also to shed
light on the importance of data pre-processing and feature engineering in achieving high prediction accuracy.

## 3.0 Exploratory data analysis

Exploratory Data Analysis (EDA) is crucial for any machine learning model. It is very helpful when gaining insights into the data to identify
patterns and trends that can be useful for the modelling process. In this study, we analysed the " NYC Taxi trip duration" dataset from Kaggle,
which contains 1,458,644 trip records with 11 different features [3]. (Information regarding each feature can be found within the appendix of
this report as expressed in a table).

By looking into the statistical summary (see appendix), there are a lot of issues regarding some of the columns. trip_duration has a maximum
value of 3526282 and a minimum value of 1 which are clearly outliers. It is also worth noting that the minimum value for passenger_count is 0
which means there are taxi rides that do not contain any passengers.

The dataset did not contain any Null or NaN values, hence there was no need to do any clean-up of these data types. The columns id and
store_and_fwd_flag do not give any useful information or correlation to our target value; hence these columns have been removed. It was
found that Vendor ID “2” take on average 213 seconds longer than Vendor ID “1”. This could be a very useful feature when training our
machine learning model. (see appendix)

By looking at specific days or parts of the day we can clearly see that there is a trend with the trip duration. Figures 1 shows the mean log of
trip_duration against the part of the day. During the early afternoon/afternoon the mean taxi trip duration is much longer than the early
morning or Night. The same could be said about Thursday and Sunday. Reasons for these include the reduction of traffic during this time and
the less active the roads are, meaning the taxi drivers can get from pickup to dropoff much faster than during the day. On Sunday, the roads
tend to be a lot quieter, meaning the time taken is considerably less.

```
Figure 1 (mean log trip_duration against pickup_day_section and pickup_day)
```
Some trips had passenger count as 0, so these rows were removed, 60 to be exact.

It was found that after calculating distance from the longatitude and latitude values, distance had a linear relationship with trip_duration,
implying that this was going to be a very important feature when we go to train our model.(see appendix)


## 4.0 Data pre-processing and Feature Selection

Data pre-processing and feature selection are critical steps in machine learning, as they help to improve the quality of the data and identify
relevant features that are predictive of the target variable [4]. In this study, we performed a comprehensive data pre-processing and feature
selection process to prepare the "Taxi trip duration" dataset for modelling.

Firstly, outliers with the original data had to be removed. To quickly remove outliers within any data, Interquartile range (IQR) [5] can be
implemented on a relevant column to remove data that does not follow any trends with most of the data. For example, time taken has a max
value of 3526282 seconds, whereas most of the data Is within the 400-1000 range. To visualise the outliers, we log scaled the trip_duration
variable and plotted it into a boxplot. Figure 2 clearly shows that there are many outliers that do not belong with most of the data. By using
IQR’s standard 25% to 75%, we can remove the outliers from our data. The result of doing IQR can be shown on the right boxplot where the
range of values of the data has been significantly lowered.

```
Figure 2 (Box plot showing range of trip_duration before and after IQR implementation)
```
For feature extraction, we calculated the distance in kilometres using the Haversine formula [6] from the longitude and latitude values. We
also reduced the dimensionality of the time and date data into more appropriate categorical data, including the month, day, part of the day,
and rush hour. We used a reference to a New York personal blog that provided us vital information about the rush hour times in New York, and
they were labelled accordingly [20].

Below is a code snippet haversine’s formula implemented as a python function.

Additionally, from the cyclical data, we converted the features into their cos and sin representations to account for the circular nature of the
data. However, after evaluating the impact of this feature on the model, the decision to not use it as the final features was made. This is
because both neural networks and random forest perform better with categorical or binary data and the performance of the models are
hindered with these types of representations [7].

When creating the distance feature through haversine’s formula, some of the distances can be classified as outliers as some distances are as
small as 0 and some distances are as large as 2000 kilometres. To remove these outliers, a threshold value for both distance and trip_duration
was made. In addition, a new column “avg_speed” was made from calculation of distance / trip_duration to calculate the average speed of the
taxi trip. An average speed of below 6 kilometers per hour and above 40 kilometres per hour was considered an outlier. (The python code for
this implementation can be found in the appendix)

After all the pre-processing, I was left with 1,240,758 rows with 7 columns. Figure 3 contains a snapshot of the dataframe in its fully cleaned
and transformed state.


```
Figure 3 (taxi dataframe displayed)
```
Overall, our data pre-processing and feature extraction process has resulted in clean and relevant features that will be used to train and test
our models.

## 5.0 Machine learning model

Using the fully pre-processed data, numerous machine learning models were trained and tested using sklearn’s test train split functionality [8].

After experimenting with various machine learning algorithms such as decision tree regression and n nearest neighbour regression, it was
found that random forest regression yielded the best accuracy and efficiency when compare to the other models. A table in the appendix
Containing the results of the different models can be found. Similarly, the same approach was done for artificial neural networks at which
alternations of sequential networks were tested to find the ideal network with n number of deep layers. The most highly performing neural
network was a sequential network with 4 layers containing dense relu activation layers, 64, 32, 16 and 1 with an optimizer of “adam”. (The
results table alongside the code snippets for each model can be found in the appendix)

### 5.1 Random Forest Regression

Random Forest Regression was the chosen model for this study. Random Forest Regression is an ensemble learning method that takes multiple
decision trees to improve the performance of the model. It randomly selects a subset of features and build a decision tree based on that
subset. The processes are repeated x number of times where the final performance score of the model is based on the average of the
predictions of all the decision trees. [9]

Random Forest Regression has several advantages over other machine learning algorithms. These advantages include its ability to handle many
features and its robustness to noise and outliers. With the fact that our dataset contains a significant amount of rows and columns, Random
Forest Regression is well-suited to identify the most important features and provide accurate predictions as well as provide the best
computational efficiency when compared to other models such as k nearest neighbour and gradient booster regression. By randomly selecting
a subset of features and building decision trees based on them, this type of regression can achieve high accuracy while avoiding overfitting to
the noise and outliers in the data [10].

Below are two code snippet taken from maching_learning.ipynb showing how to train and test a Random Forest Regression model and
displaying the results of the model.

### 5.2 Artificial Neural Network

An Artificial Neural Network (ANN) is a machine learning algorithm that draws inspiration from the structure and function of the human brain.
Much like biological neural networks, ANNs consist of interconnected nodes or units, referred to as neurons [11]. These neurons are arranged
in layers, allowing the network to process and learn complex relationships between input and output variables.


Artificial Neural Networks offer several advantages over other machine learning algorithms, making them an ideal choice for various
applications. These advantages include their ability to handle nonlinear relationships and their robustness to noise and outliers. Given that the
taxi dataset comprises a considerable number of rows and columns, ANNs are well-suited for portraying the underlying patterns and providing
accurate predictions. Using multiple hidden layers and nonlinear activation functions, ANNs can achieve high accuracy without overfitting to
noise and outliers in the data.

Below are two code snippet taken from deep_learning.ipynb showing how the testing and training of an artificial neural network and
displaying the results of the model.

## 6.0 Evaluation....................................................................................................................................................................

The performance metrics used in this study are explained below.

MSE – How close a regression line is to a set of points; it is calculated by taking the distances from the points to the regression line and
squaring them [12].

R2 – An evaluation measure for regression-based machine models. It is also known as the coefficient of determination. It is the most important
metric that can be used to evaluate the performance of a regression model. It is a calculation of the difference between the true values in the
dataset and the predictions made by the model [13].

Before we compare the performance of the two models, it is important that we evaluate each model by tweaking their hyperparameters and
apply cross validation testing. Hyperparameters are certain requirements that can change the performance and efficiency of a model. A model
can perform very poorly if these values are not configured correctly. To deploy hyperparameter tweaking, we can iterate through different
hyperparameter values and test and train a model with each different value. After the iterations, we can plot the performances against the
parameter values, and we identify the optimum hyperparameter value.

### 6.1 Machine Learning Hyperparameters

For the machine learning model, we investigate the hyperparamters ‘n_estimators’ and ‘depth’

The number of estimators parameter is responsible for the number of individual decision trees the model will create. For example, if the model
has n_estimators = 10 then the model will have 10 decision trees. Figure 4 reveals the results of iterations of n_estimators and its affect on R
and MSE scores (code can be found in the appendix). The model is performing at its best when the r2 score is at its maximum and when the
mse is at the minimum. In this case, this value is 100, therefore the best value for n_estimators is 100.


```
Figure 4 (Hyper parameter number of estimators against score metrics)
```
Another hyperparameter called depth was also investigated. Depth is referred as the maximum number of levels in each decision tree that
makes up the forest. Increasing the depth over a certain value can cause overfitting, therefore careful tweaking of this value is required to fully
optimise the model.

Figure 5 shows the results of increments of 1 for the depth. The graphs show the effect on r2 and mse scores. As clearly shown in the graph,
the optimum depth value is 10 as r2 score is at its highest and mse is at its lowest.

```
Figure 5 (Hyper parameter depth against score metrics)
```
### 6.1 Neural Network Hyperparameters

For the neural network model, we investigate the hyperparameters ‘epochs and ‘batch_size’.

An epoch is defined as the number of times the dataset is passed through the training process of a neural network. The effect of the epoch size
on the performance of ANN models is investigated by training several models with different epoch values for a given dataset.

Figures 6 visualises the iteration process by plotting a line of best fit through each iteration point. The R2 score and MSE are plotted against
each iteration. The graphs show a very skewed performance at which the model did the best at approximately epoch =7.

```
Figure 6 (Hyper parameter epochs against score metrics)
```
Batch size refers to the number of training samples that are processed at each iteration of the model’s training sequence. By dividing the data
into batches, the model can update its weights more frequently and see more samples in each epoch. This significantly improves the
processing time per epoch as there is much less data to process. Having a smaller value for batch size results in a longer processing time as the
number refers to how many divisions the original data is being split into.


This hyperparameter was also tested through a range of 2 to 250, at which the most optimum value was batch_size = 16.

```
Figure 7 (Hyper parameter batch size against score metrics)
```
### 6.2 K-Fold Cross-Validation

Now that the models have had their hyperparameters tweaked, it is time to now compare the performances of the two models. However,
before we compare the performances, we must apply k-fold cross validation on both models to see their genuine performance. Cross-
validation using k-fold is a popular machine learning technique that involves the splitting of the dataset into k equal-sized folds [14]. An
average of the performance of each K fold is then taken to determine the overall performance of the model. As a result, this process provides a
much more reliable assessment of the performance of any machine learning model. By only relying on one test-train split for the model, the
performance of the model may be highly dependent on that instance of a single test-train split. This may lead to overfitting whereby the model
performs worse on unseen data. Therefore the need for cross validation is very important for any machine learning model. The results of k-fold
cross validation of both models can be shown in figures 8 and 9.

```
Figure 8 (K-fold cross validation on neural network)
```
```
Figure 9 (K-fold cross validation on Random Forest)
```

```
Table 1 (Performance metrics from the k-fold cross validation)
```
```
Model Mean R2 Score Mean MSE Score
Artificial Neural Network 0.7723 0.
Random Forest Regression 0.7746 0.
```
It was found that Random Forest Regression performs slightly better than the artificial neural network as it has a higher r2 and lower mse
score. The IQR for r2 score for Random Forest was significantly smaller than the neural network. This could be a good indicator that the model
is performing well in a set range of r2 scores, rather the neural network is more skewed in terms of average performance.

Figure 10 (True vs Random Forest predictions)

```
To conclude, a plot of the best performing model of true vs predicted
is shown below. Showing a linear relationship between the true values
and Random Forest’s predictions. The graph reveals that the model
can accurately predict the trip duration of a taxi rides in New York City.
Therefore, giving the potential use and deployment for transportation
companies to help optimise their services and give useful information
to customers.
```
### 6.3 Limitations

One limitation of using artificial neural networks is the computational complexity. ANNs use up a lot of resources and take a very long time to
train. A reason for this is because ANNs require many iterations that are used to optimise the weights of the model which get more
computationally expensive for larger datasets. This is especially true for our dataset as it contains 1.2 million rows (after pre-processing).
Random forest was a better choice as it was much faster and computationally efficient than an ANN.

Some other limitations of the models include the lack of useful features such as weather conditions, traffic jams, road works and special
events. These types of features can be difficult to measure and/or predict in real time, making it challenging to implement into a machine
learning model. All real-world factors that can heavily affect the model’s performance should be included with the model to help improve
accuracy and help deal more with reasonings for why some taxi trips took longer than they should have.

In addition, the model may not perform well with places outside New York, as the data has been specifically trained on information that is
taken from New York City. The model may fail to generalise other locations where there might be different factors that can affect the trip
duration of the taxi.

## 7.0 Conclusion, future work recommendations

After hyperparameter tweaking, both the machine learning and neural network models performed similarly to each other, but Random Forest
yielded better results in terms of performance metrics and computational cost. Random Forest processed significantly faster than ANN,
resulting in a more robust and efficient model.

Further work in this area could include the implementation of geo mapping [15] the longitude and latitude values to get a better
representation of where the locations are in New York. Pairing this with Dijkstra's shortest path algorithm [16] could be used to find the
optimum route from pickup to drop-off location. Currently, the distance is calculated as a single vector, whereas the implementation of
remapping alongside Dijkstra's algorithm will significantly improve the distance calculation by considering pathing routes instead of a straight
line.

In conclusion, our model has shown promising results in predicting taxi trip durations in New York City. Future work could focus on
implementing additional features such as weather conditions, traffic jams, and special events to improve accuracy further. Kaggle offers a
dataset that contains the weather information for New York in 2016 [17]. This means that both datasets can be merged to create new features
for the machine learning model to further improve the performance.


## 8.0 references

[1] Zhao, Z., Chi, Y., Ding, Z., Chang, M. and Cai, Z. (2023). Latent Semantic Sequence Coding Applied to Taxi Travel Time
Estimation. ISPRS International Journal of Geo-Information, [online] 12(2), p.44. Available at: https://www.mdpi.com/2220-
9964/12/2/44/pdf [Accessed 23 Mar. 2023].

[2] MIT Press. (n.d.). Fundamentals of Machine Learning for Predictive Data Analytics. [online] Available at:
https://mitpress.mit.edu/9780262029445/fundamentals-of-machine-learning-for-predictive-data-analytics/ [Accessed 2 Mar.
2023].

[3] kaggle.com. (n.d.). New York City Taxi Trip Duration. [online] Available at: https://www.kaggle.com/competitions/nyc-taxi-
trip-duration [Accessed 23 Mar. 2023].

[4] KDnuggets. (n.d.). Importance of Pre-Processing in Machine Learning. [online] Available at:
https://www.kdnuggets.com/2023/02/importance-preprocessing-machine-learning.html [Accessed 23 Mar. 2023].

[5] Wikipedia Contributors (2019). Interquartile range. [online] Wikipedia. Available at:
https://en.wikipedia.org/wiki/Interquartile_range.

[6] Wikipedia Contributors (2019). Haversine formula. [online] Wikipedia. Available at:
https://en.wikipedia.org/wiki/Haversine_formula.

[7] Stack Overflow. (n.d.). machine learning - Cyclic ordinal features in random forest. [online] Available at:
https://stackoverflow.com/questions/47350569/cyclic-ordinal-features-in-random-forest [Accessed 23 Mar. 2023].

[8] Scikit-learn.org. (2018). sklearn.model_selection.train_test_split — scikit-learn 0.20.3 documentation. [online] Available at:
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html.

[9] Team, G.L. (2020). Random Forest Algorithm- An Overview | Understanding Random Forest. [online] GreatLearning. Available
at: https://www.mygreatlearning.com/blog/random-forest-algorithm/.

[10] Ellis, C. (2021). Random forest overfitting. [online] Crunching the Data. Available at: https://crunchingthedata.com/random-
forest-overfitting/#:~:text=In%20general%2C%20random%20forests%20are [Accessed 23 Mar. 2023].

[11] Analytics Vidhya. (2021). Artificial Neural Network | How does Artificial Neural Network Work. [online] Available at:
https://www.analyticsvidhya.com/blog/2021/04/artificial-neural-network-its-inspiration-and-the-working-mechanism/.

[12] Statistics How To. (n.d.). Mean Squared Error: Definition and Example. [online] Available at:
https://www.statisticshowto.com/probability-and-statistics/statistics-definitions/mean-squared-error/.

[13] Kharwal, A. (2021). R2 Score in Machine Learning. [online] Data Science | Machine Learning | Python | C++ | Coding |
Programming | JavaScript. Available at: https://thecleverprogrammer.com/2021/06/22/r2-score-in-machine-learning/.

[14] SciKit-Learn (2009). 3.1. Cross-validation: evaluating estimator performance — scikit-learn 0.21.3 documentation. [online]
Scikit-learn.org. Available at: https://scikit-learn.org/stable/modules/cross_validation.html.

[15] Stewart, R. (2018). GeoPandas 101: Plot any data with a latitude and longitude on a map. [online] Medium. Available at:
https://towardsdatascience.com/geopandas-101-plot-any-data-with-a-latitude-and-longitude-on-a-map-98e01944b972.

[16] freeCodeCamp.org. (2020). Dijkstra’s Shortest Path Algorithm - A Detailed and Visual Introduction. [online] Available at:
https://www.freecodecamp.org/news/dijkstras-shortest-path-algorithm-visual-introduction/#:~:text=Dijkstra.

[17] [http://www.kaggle.com.](http://www.kaggle.com.) (n.d.). Weather data in New York City - 2016. [online] Available at:
https://www.kaggle.com/datasets/mathijs/weather-data-in-new-york-city-2016 [Accessed 23 Mar. 2023].

[18] IBM (n.d.). What is Machine Learning? | IBM. [online] [http://www.ibm.com.](http://www.ibm.com.) Available at: https://www.ibm.com/uk-
en/topics/machine-learning.

[19] The Editors of Encyclopedia Britannica (2019). latitude and longitude | Description & Diagrams. In: Encyclopædia Britannica.
[online] Available at: https://www.britannica.com/science/latitude.

[20] Knispel, J. (2022). Worst Traffic Times in New York City. [online] Law Offices of Jay S. Knispel Personal Injury Lawyers.
Available at: https://jknylaw.com/blog/worst-traffic-times-in-new-york-city/.


## Appendix

Deep Learning files located on google colab

https://colab.research.google.com/drive/14-MtJZrDjMBvq5m6R2PeFYfcfpJ6m7ej?usp=sharing

Link to fully preprocessed data as well as the raw data

https://livereadingac-
my.sharepoint.com/:f:/g/personal/jz015642_student_reading_ac_uk/EnXpjL79KDdOueocVjuIiRUBURU4A69QPmSITLVVrAoO7g?
e=3TdCRB

## 3.


Table showing the features with their relevant description.

```
id a unique identifier for each trip
vendor_id a code indicating the provider associated with the trip record
pickup_datetime date and time when the meter was engaged
dropoff_datetime date and time when the meter was disengaged
passenger
Pickup_longitude the longitude where the meter was engaged
Pickup_latitude the latitude where the meter was engaged
dropoff_longitude
the longitude where the meter was disengaged
```
```
dropoff_latitude the latitude where the meter was disengaged
store_and_fwd_flag This flag indicates whether the trip record was held in vehicle memory before sending to the vendor because
the vehicle did not have a connection to the server - Y=store and forward; N=not a store and forward trip
trip_duration duration of the trip in seconds
```

- 1.0 Abstract
- 2.0 Background
- 3.0 Exploratory data analysis
- 4.0 Data pre-processing and Feature Selection
- 5.0 Machine learning model
   - 5.1 Random Forest Regression
   - 5.2 Artificial Neural Network
- 6.0 Evaluation....................................................................................................................................................................
   - 6.1 Machine Learning Hyperparameters
   - 6.1 Neural Network Hyperparameters
   - 6.2 K-Fold Cross-Validation
   - 6.3 Limitations
- 7.0 Conclusion, future work recommendations
- 8.0 references
- Appendix
   - 3.0
   - 4.0
   - 5.0
   - 6.0
   - Definitions and equations
- 4.




## 5.

Machine learning table

```
Model R2 score mse
Random Forest
Regression
```
#### 0.7748166735174706 0.

```
Decision tree
regression
```
#### 0.7728617024124562 0.

```
N nearest
neighbour
regression
```
#### 0.7334185864182337 0.

```
Ada booster
regression
```
#### 0.7529829809524184 0.


```
Gradient booster
regression
```
#### 0.7758346869884016 0.

Machine learning code


Artificial neural network table

```
Model Number
of
layers
```
```
Layer R2 score (3.sf) MSE (3.sf)
```
```
Sequential 6 (Dense(64, activation='relu', input_dim=X_train.shape[1]))
(Dense(32, activation='relu'))
(Dense(16, activation='relu'))
(Dense(8, activation='relu'))
(Dense(4, activation='relu'))
(Dense(1, activation='linear'))
```
#### 0.773 0.

```
Sequential 2 (Dense(64, activation='relu', input_dim=X_train.shape[1]))
(Dense(1, activation='linear')
```
#### 0.766 0.

```
Sequential 3 (Dense(64, activation='relu', input_dim=X_train.shape[1]))
(Dense(32, activation='relu'))
(Dense(1, activation='linear'))
```
#### 0.774 0.

```
Sequential 4
(Dense(64, activation='relu', input_dim=X_train.shape[1]))
```
```
(Dense(32, activation='relu'))
(Dense(16, activation='relu'))
(Dense(1, activation='linear'))
```
#### 0.775 0.

```
Sequential 5 (Dense(64, activation='relu',
input_dim=X_train.shape[1]))
```
```
(Dense(32, activation='relu'))
```
```
(Dense(16, activation='relu'))
```
```
(Dense(8, activation='relu'))
```
```
(Dense(1, activation='linear'))
```
#### 0.772 0.

```
Sequential 7
(Dense(64, activation='relu', input_dim=X_train.shape[1]))
```
```
(Dense(32, activation='relu'))
(Dense(16, activation='relu'))
(Dense(8, activation='relu'))
(Dense(4, activation='relu'))
(Dense(2, activation='relu'))
```
```
-7.87e-06 0.
```

```
(Dense(1, activation='linear'))
```
ANN code




### 6.0






### Definitions and equations

latitude and longitude, coordinate system by means of which the position or location of any place on Earth’s surface can be determined and
described. [19]


The haversine formula determines the great-circle distance between two points on a sphere given their longitudes and latitudes. Important in
navigation, it is a special case of a more general formula in spherical trigonometry, the law of haversines, that relates the sides and angles of
spherical triangles. [6]

Machine learning is a branch of artificial intelligence (AI) and computer science which focuses on the use of data and algorithms to imitate the
way that humans learn, gradually improving its accuracy. [19]

```
Model Numbe
r of
layers
```
```
Layer R2 score (3.sf) MSE (3.sf) RMS
E
```
```
Sequentia
l
```
```
6 (Dense(64, activation='relu', input_dim=X_train.shape[1
]))
(Dense(32, activation='relu'))
(Dense(16, activation='relu'))
(Dense(8, activation='relu'))
(Dense(4, activation='relu'))
(Dense(1, activation='linear'))
```
#### 0.775 0.0923 0.304

```
Sequentia
l
```
```
2 (Dense(64, activation='relu', input_dim=X_train.shape[1
]))
(Dense(1, activation='linear')
```
#### 0.766 0.0965 0.312

```
Sequentia
l
```
```
3 (Dense(64, activation='relu', input_dim=X_train.shape[1
]))
(Dense(32, activation='relu'))
(Dense(1, activation='linear'))
```
#### 0.7738138207655503 0.0932722277217959

#### 7

```
Sequentia
l
```
#### 4

```
(Dense(64, activation='relu', input_dim=X_train.shape[1
]))
```
```
(Dense(32, activation='relu'))
(Dense(16, activation='relu'))
(Dense(1, activation='linear'))
```
#### 0.7746210813716057 0.0929393382617340

#### 2

```
Sequentia
l
```
```
5 (Dense(64, activation='relu',
input_dim=X_train.shape[1]))
```
#### 0.7726110395953949 0.0937682177048352

#### 9


```
(Dense(32, activation='relu'))
```
```
(Dense(16, activation='relu'))
```
```
(Dense(8, activation='relu'))
```
(Dense(1, activation='linear'))
Sequentia
l

#### 7

```
(Dense(64, activation='relu', input_dim=X_train.shape[1
]))
```
```
(Dense(32, activation='relu'))
(Dense(16, activation='relu'))
(Dense(8, activation='relu'))
(Dense(4, activation='relu'))
(Dense(2, activation='relu'))
(Dense(1, activation='linear'))
```
#### -

```
7.873200537300562e
-06
```
#### 0.4123725082957821

Model R2 score mse
Random Forest
Regression

#### 0.7748166735174706 0.09285868207296308

Decision tree
regression

#### 0.7728617024124562 0.09366485206404565

N nearest
neighbour
regression

#### 0.7334185864182337 0.10992998068296528

Ada booster
regression

#### 0.7529829809524184 0.10186222575466765

Gradient booster
regression

#### 0.7758346869884016 0.09243888460962599


Random forest max depth variations, peak is max depth =10

Log mean trip duration against different parts of the day


Afternoon is where the most is

Log mean trip duration against days of the week

Thursday’s seem to have the most trip durations and Sunday seem to have the least

Log mean trip duration against month of pickup

June and may seem to have significantly longer trip durations whereas January have significantly lower trip durations. Unfortunately, data
after June does not exist, therefore this feature can only be useable if the date is within January to June


Log mean trip duration against rush hour

Clearly, rush hour affects the trip duration... rush hour in the afternoon has much higher trip duration than rush hour in the morning.




#### ANN

For epochs

Best epochs = 7

For batch size


Best Batch size = 16

K fold for cnn with 64 32 16 1

4 layers

Mean R2 score: 0.7723111281968238
Mean MSE: 0.09337855344673832

K fold for random forest 100 n estimators 10 depth


Mean R2 score: 0.7746809624274605
Mean MSE: 0.09240789871753803


