# Pyspark ALS Model Applications
Notes here are from a Course in [DataCamp](https://campus.datacamp.com/courses/recommendation-engines-in-pyspark).

## Recommendation System with PySpark

### Toy 1
Suppose we have pyspark dataframe called ratings:
```bash
In [1]: ratings.show(5)
+------+-------+------+
|userId|movieId|rating|
+------+-------+------+
|     2|      3|   3.0|
|     2|      1|   4.0|
|     2|      2|   4.0|
|     2|      0|   3.0|
|     0|      3|   4.0|
+------+-------+------+
only showing top 5 rows
```

```python
# Split the ratings dataframe into training and test data
(training_data, test_data) = ratings.randomSplit([0.8, 0.2], seed=42)

# Set the ALS hyperparameters
from pyspark.ml.recommendation import ALS
als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating", rank =10, maxIter =15, regParam =0.1,
          coldStartStrategy="drop", nonnegative =True, implicitPrefs = False)

# Fit the mdoel to the training_data
model = als.fit(training_data)

# Generate predictions on the test_data
test_predictions = model.transform(test_data)
test_predictions.show()

# Import RegressionEvaluator
from pyspark.ml.evaluation import RegressionEvaluator

# Complete the evaluator code
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")

# Extract the 3 parameters
print(evaluator.getMetricName())
print(evaluator.getLabelCol())
print(evaluator.getPredictionCol())

# Evaluate the "test_predictions" dataframe
RMSE = evaluator.evaluate(test_predictions)

# Print the RMSE
print (RMSE)
```

### Toy2
```python
# Look at the column names
print(ratings.columns)

# Look at the first few rows of data
print(ratings.show())
```

The output is like this:
```bash 
['userId', 'movieId', 'rating', 'timestamp']
    +------+-------+------+----------+
    |userId|movieId|rating| timestamp|
    +------+-------+------+----------+
    |     1|     31|   2.5|1260759144|
    |     1|   1029|   3.0|1260759179|
    |     1|   1061|   3.0|1260759182|
    |     1|   1129|   2.0|1260759185|
    |     1|   1172|   4.0|1260759205|
    |     1|   1263|   2.0|1260759151|
    |     1|   1287|   2.0|1260759187|
    |     1|   1293|   2.0|1260759148|
    |     1|   1339|   3.5|1260759125|
    |     1|   1343|   2.0|1260759131|
    |     1|   1371|   2.5|1260759135|
    |     1|   1405|   1.0|1260759203|
    |     1|   1953|   4.0|1260759191|
    |     1|   2105|   4.0|1260759139|
    |     1|   2150|   3.0|1260759194|
    |     1|   2193|   2.0|1260759198|
    |     1|   2294|   2.0|1260759108|
    |     1|   2455|   2.5|1260759113|
    |     1|   2968|   1.0|1260759200|
    |     1|   3671|   3.0|1260759117|
    +------+-------+------+----------+
    only showing top 20 rows
```

Calculate Sparsity:

```python
# Count the total number of ratings in the dataset
numerator = ratings.select("rating").count()

# Count the number of distinct userIds and distinct movieIds
num_users = ratings.select("userId").distinct().count()
num_movies = ratings.select("movieId").distinct().count()

# Set the denominator equal to the number of users multiplied by the number of movies
denominator = num_users * num_movies

# Divide the numerator by the denominator
sparsity = (1.0 - (numerator *1.0)/denominator)*100
print("The ratings dataframe is ", "%.2f" % sparsity + "% empty.")
```

```bash
The ratings dataframe is  98.36% empty.
```

Explore the dataset:
```python
# Import the requisite packages
from pyspark.sql.functions import col

# View the ratings dataset
ratings.show()

# Filter to show only userIds less than 100
ratings.filter(col("userId") < 100).show()

# Group data by userId, count ratings
ratings.groupBy("userId").count().show()
```


