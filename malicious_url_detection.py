# Distributed Malicious URL Detection with Spark ML
#
# Curated from the original coursework notebook (submitted as a PDF export).
# Covers: data loading, class balancing, text cleaning, TF-IDF feature
# extraction for both URL and page content, K-means clustering, feature
# assembly, and classification with Random Forest, Logistic Regression and
# Decision Tree models, including hyperparameter tuning. Teammates' names,
# student IDs and personal file paths have been removed. This is a cleaned,
# representative excerpt rather than a line-for-line copy of every cell in
# the original notebook.

import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, rand, size, split, udf, lower, regexp_replace, monotonically_increasing_id,
)
from pyspark.sql.types import StringType
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    Tokenizer, StopWordsRemover, HashingTF, IDF, VectorAssembler, StringIndexer,
)
from pyspark.ml.clustering import KMeans
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

spark = SparkSession.builder.appName("courseWork").config("spark.memory.fraction", 0.8).getOrCreate()

# ---------------------------------------------------------------------------
# Task 1 - Data loading and preprocessing
# ---------------------------------------------------------------------------
df_raw_data = (
    spark.read.option("multiLine", "true").option("escape", '"')
    .csv("data/Webpages_Classification_train_data.csv", inferSchema=True, header=True, sep=",")
)
df_raw_data_intermediate = df_raw_data.drop("_c0")

# Balance the 'good' / 'bad' classes by downsampling the majority class
label_count = df_raw_data_intermediate.groupBy("label").count()
underrepresented_count = label_count.where(label_count["label"] == "bad").collect()[0]["count"]
total_count = df_raw_data_intermediate.count()

df_raw_balanced_data_untrimmed = (
    df_raw_data_intermediate.filter(df_raw_data_intermediate["label"] == "good")
    .sample(False, underrepresented_count / total_count)
)
df_raw_balanced_data_untrimmed = df_raw_balanced_data_untrimmed.union(
    df_raw_data_intermediate.filter(df_raw_data_intermediate["label"] == "bad")
)

# Drop very short content entries, then shuffle
df_raw_balanced_data_with_word_count = df_raw_balanced_data_untrimmed.withColumn(
    "word_count", size(split(df_raw_balanced_data_untrimmed["content"], " "))
)
df_raw_balanced_data_trimmed = df_raw_balanced_data_with_word_count.filter(
    df_raw_balanced_data_with_word_count["word_count"] >= 60
).drop("word_count")

balanced_data_with_random = df_raw_balanced_data_trimmed.withColumn("rand", rand())
df_raw_data_file_final = balanced_data_with_random.orderBy("rand").drop("rand")

# ---------------------------------------------------------------------------
# Text cleaning + TF-IDF feature extraction (URL and page content)
# ---------------------------------------------------------------------------
stop = stopwords.words("english")


def remove_stopwords(text):
    return " ".join([word for word in text.split() if word not in stop])


remove_stopwords_udf = udf(remove_stopwords, StringType())

df_raw_data_file_final = df_raw_data_file_final.withColumn(
    "content", lower(regexp_replace("content", r"[^\w\s]", ""))
)
df_raw_data_file_final = df_raw_data_file_final.withColumn("content", remove_stopwords_udf("content"))
df_raw_data_file_final = df_raw_data_file_final.withColumn(
    "url", lower(regexp_replace("url", r"[^\w\s]", ""))
)
df_raw_data_file_final = df_raw_data_file_final.withColumn("url", remove_stopwords_udf("url"))

tokenizer = Tokenizer(inputCol="url", outputCol="url_words")
tokenizer2 = Tokenizer(inputCol="content", outputCol="content_words")

stopwords_remover = StopWordsRemover(inputCol="url_words", outputCol="filtered_url_words")
stopwords_remover2 = StopWordsRemover(inputCol="content_words", outputCol="filtered_content_words")

hashingTF = HashingTF(inputCol="filtered_url_words", outputCol="url_tf_features", numFeatures=8000)
hashingTF2 = HashingTF(inputCol="filtered_content_words", outputCol="content_tf_features", numFeatures=8000)

idf = IDF(inputCol="url_tf_features", outputCol="url_tfidf_features")
idf2 = IDF(inputCol="content_tf_features", outputCol="content_tfidf_features")

pipeline_url = Pipeline(stages=[tokenizer, stopwords_remover, hashingTF, idf])
pipeline_model_url = pipeline_url.fit(df_raw_data_file_final)
url_tfidf = pipeline_model_url.transform(df_raw_data_file_final)

pipeline_content = Pipeline(stages=[tokenizer2, stopwords_remover2, hashingTF2, idf2])
pipeline_model_content = pipeline_content.fit(df_raw_data_file_final)
content_tfidf = pipeline_model_content.transform(df_raw_data_file_final)

# ---------------------------------------------------------------------------
# Clustering similar URLs / content (K-means, elbow method for k)
# ---------------------------------------------------------------------------
assembler_url = VectorAssembler(inputCols=["url_tfidf_features"], outputCol="features_url")
assembler_content = VectorAssembler(inputCols=["content_tfidf_features"], outputCol="features_content")
url_tfidf_assembled = assembler_url.transform(url_tfidf)
content_tfidf_assembled = assembler_content.transform(content_tfidf)


def calculate_wcss(data, feature_col, k_max):
    wcss = []
    for k in range(2, k_max + 1):
        kmeans = KMeans(featuresCol=feature_col, k=k)
        model = kmeans.fit(data)
        wcss.append(model.summary.trainingCost)
    return wcss


wcss_url = calculate_wcss(url_tfidf_assembled, "features_url", 10)
plt.plot(range(2, 11), wcss_url)
plt.title("Elbow Method for URL TF-IDF Features")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()

num_clusters_url = 4
num_clusters_content = 5
model_url = KMeans(featuresCol="features_url", k=num_clusters_url, seed=123).fit(url_tfidf_assembled)
model_content = KMeans(featuresCol="features_content", k=num_clusters_content, seed=123).fit(content_tfidf_assembled)
clustered_url = model_url.transform(url_tfidf_assembled)
clustered_content = model_content.transform(content_tfidf_assembled)

# Join cluster assignments back onto the main DataFrame
df_raw_data_file_final = df_raw_data_file_final.withColumn("id", monotonically_increasing_id())
clustered_url_with_id = clustered_url.withColumn("id", monotonically_increasing_id()) \
    .select("id", col("prediction").alias("url_prediction"))
clustered_content_with_id = clustered_content.withColumn("id", monotonically_increasing_id()) \
    .select("id", col("prediction").alias("content_prediction"))

df_raw_data_file_final = df_raw_data_file_final.join(clustered_content_with_id, "id", "inner")
df_raw_data_file_final = df_raw_data_file_final.join(clustered_url_with_id, "id", "inner")

# ---------------------------------------------------------------------------
# Task 2 - Feature assembly and model training (Random Forest, Logistic
# Regression, Decision Tree)
# ---------------------------------------------------------------------------
feature_columns = [
    "url_prediction", "url_len", "geo_loc", "tld", "who_is", "https",
    "content_prediction", "js_len", "js_obf_len",
]
categorical_cols = ["geo_loc", "tld", "who_is", "https"]

indexers = [
    StringIndexer(inputCol=c, outputCol=c + "_index", handleInvalid="keep") for c in categorical_cols
]
df_indexed_cats = Pipeline(stages=indexers).fit(df_raw_data_file_final).transform(df_raw_data_file_final)

assembler = VectorAssembler(
    inputCols=[c + "_index" if c in categorical_cols else c for c in feature_columns],
    outputCol="features",
)
df_with_features = assembler.transform(df_indexed_cats)

label_indexer = StringIndexer(inputCol="label", outputCol="indexed_label")
df_indexed = label_indexer.fit(df_with_features).transform(df_with_features)

train_data, test_data = df_indexed.randomSplit([0.7, 0.3], seed=42)

# --- Random Forest (scikit-learn, run on collected features) ---
X_train = np.array([row.features.toArray() for row in train_data.select("features").collect()])
y_train = np.array([row.indexed_label for row in train_data.select("indexed_label").collect()])
rf_classifier = RandomForestClassifier(n_estimators=100, criterion="gini", random_state=42)
rf_classifier.fit(X_train, y_train)

X_test = np.array([row.features.toArray() for row in test_data.select("features").collect()])
y_pred_rf = rf_classifier.predict(X_test)

# --- Logistic Regression (Spark ML) ---
lr = LogisticRegression(featuresCol="features", labelCol="indexed_label")
lr_model = lr.fit(train_data)

# --- Decision Tree (Spark ML) ---
decision_tree_classifier = DecisionTreeClassifier(
    labelCol="indexed_label", featuresCol="features", maxBins=500
)
decision_tree_model = Pipeline(stages=[decision_tree_classifier]).fit(train_data)
predictions = decision_tree_model.transform(test_data)

evaluator = MulticlassClassificationEvaluator(
    labelCol="indexed_label", predictionCol="prediction", metricName="accuracy"
)
accuracy = evaluator.evaluate(predictions)
print("Decision Tree accuracy:", accuracy)
# Observed accuracy on the supplied dataset: ~0.9985 (99.8%)

# ---------------------------------------------------------------------------
# Task 3 - Hyperparameter tuning
# ---------------------------------------------------------------------------
# Random Forest: grid search over n_estimators / criterion
param_grid = [{"n_estimators": list(range(10, 120, 10)), "criterion": ["gini", "entropy"]}]
grid = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=5)
grid.fit(X_train, y_train)
best_params, best_estimator, best_score = grid.best_params_, grid.best_estimator_, grid.best_score_
y_pred_best_rf = best_estimator.predict(X_test)

# Logistic Regression: cross-validated grid search over regularization params
lr_param_grid = (
    ParamGridBuilder()
    .addGrid(lr.regParam, [0.1, 0.01, 0.001])
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
    .addGrid(lr.maxIter, [10, 50, 100])
    .build()
)
cv = CrossValidator(
    estimator=lr,
    estimatorParamMaps=lr_param_grid,
    evaluator=MulticlassClassificationEvaluator(labelCol="indexed_label"),
    numFolds=5,
)
cv_model = cv.fit(train_data)
best_lr_model = cv_model.bestModel
