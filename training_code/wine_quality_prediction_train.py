from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.mllib.evaluation import MulticlassMetrics

def preprocess_data(input_df):
    return input_df.select(*(col(col_name).cast("double").alias(col_name.strip("\"")) for col_name in input_df.columns))

if __name__ == "__main__":
    # Initialize Spark session
    spark = SparkSession.builder.appName('wine_quality_prediction').getOrCreate()
    spark.sparkContext.setLogLevel('ERROR')

    # File paths on AWS S3
    input_path_train = "s3://sg2489-wqp/TrainingDataset.csv"
    input_path_valid = "s3://sg2489-wqp/ValidationDataset.csv"
    output_path = "s3://sg2489-wqp/trained_model.model"

    # Read and preprocess training data
    df_train = spark.read.format("csv").option('header', 'true').option("sep", ";").option("inferschema",'true').load(input_path_train)
    train_data = preprocess_data(df_train)

    # Read and preprocess validation data
    df_valid = spark.read.format("csv").option('header', 'true').option("sep", ";").option("inferschema",'true').load(input_path_valid)
    valid_data = preprocess_data(df_valid)

    # Define feature columns
    feature_columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']

    # Assemble features and index label
    feature_assembler = VectorAssembler(inputCols=feature_columns, outputCol='features')
    label_indexer = StringIndexer(inputCol="quality", outputCol="label")

    train_data.cache()
    valid_data.cache()

    # Random Forest Classifier configuration
    random_forest = RandomForestClassifier(labelCol='label', featuresCol='features', numTrees=150, maxBins=8, maxDepth=15, seed=150, impurity='gini')
    pipeline = Pipeline(stages=[feature_assembler, label_indexer, random_forest])
    model = pipeline.fit(train_data)

    # Evaluate the initial model
    predictions = model.transform(valid_data)
    evaluator = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction', metricName='accuracy')
    accuracy = evaluator.evaluate(predictions)
    print('Test Accuracy of Initial Model:', accuracy)

    # Compute weighted f1 score of the initial model
    metrics = MulticlassMetrics(predictions.select(['prediction', 'label']).rdd.map(tuple))
    print('Weighted f1 score of Initial Model:', metrics.weightedFMeasure())

    # Define parameter grid for cross-validation
    param_grid = ParamGridBuilder() \
        .addGrid(random_forest.maxBins, [8, 4]) \
        .addGrid(random_forest.maxDepth, [25, 6]) \
        .addGrid(random_forest.numTrees, [500, 50]) \
        .addGrid(random_forest.minInstancesPerNode, [6]) \
        .addGrid(random_forest.seed, [100, 200]) \
        .addGrid(random_forest.impurity, ["entropy", "gini"]) \
        .build()

    # Cross-validation configuration
    cross_validator = CrossValidator(estimator=pipeline, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=2)
    cv_model = cross_validator.fit(train_data)

    # Retrieve the best model from cross-validation
    best_model = cv_model.bestModel
    print('Best Model:', best_model)

    # Evaluate the best model
    predictions_best = best_model.transform(valid_data)
    accuracy_best = evaluator.evaluate(predictions_best)
    print('Test Accuracy of Best Model:', accuracy_best)

    # Compute weighted f1 score of the best model
    metrics_best = MulticlassMetrics(predictions_best.select(['prediction', 'label']).rdd.map(tuple))
    print('Weighted f1 score of Best Model:', metrics_best.weightedFMeasure())

    # Save the best model
    best_model.write().overwrite().save(output_path)