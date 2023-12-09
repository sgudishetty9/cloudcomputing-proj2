import os
import sys
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.functions import col

def process_input_dataframe(input_dataframe):
    return input_dataframe.select(*(col(col_name).cast("double").alias(col_name.strip("\"")) for col_name in input_dataframe.columns))

if __name__ == "__main__":
    spark_session = SparkSession.builder \
        .appName('wine_quality_predictor_spark_app') \
        .getOrCreate()
    spark_session.sparkContext.setLogLevel('ERROR')

    current_working_dir = os.getcwd()

    # Handle input arguments for data and model paths
    if len(sys.argv) > 3:
        sys.exit(-1)
    elif len(sys.argv) > 1:
        input_data_path = sys.argv[1]
        if not ("/" in input_data_path):
            input_data_path = os.path.join(current_working_dir, input_data_path)
        model_location = os.path.join(current_working_dir, "trained_model")
        print("Test data file location:")
        print(input_data_path)
    else:
        print("Current directory:")
        print(current_working_dir)
        input_data_path = os.path.join(current_working_dir, "ValidationDataset.csv")
        model_location = os.path.join(current_working_dir, "trained_model")

    # Read CSV file into DataFrame
    input_df = (spark_session.read
                .format("csv")
                .option('header', 'true')
                .option("sep", ";")
                .option("inferschema", 'true')
                .load(input_data_path))

    cleaned_input_df = process_input_dataframe(input_df)

    # Define features for the model
    selected_model_features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']

    # Load the pre-trained model
    trained_ml_model = PipelineModel.load(model_location)

    # Make predictions on the data
    predictions_result = trained_ml_model.transform(cleaned_input_df)
    print("Sample Predictions:")
    print(predictions_result.show(5))

    # Evaluate the model
    evaluation_results = predictions_result.select(['prediction', 'label'])
    evaluation_metric = MulticlassClassificationEvaluator(
        labelCol='label',
        predictionCol='prediction',
        metricName='accuracy'
    )
    model_accuracy = evaluation_metric.evaluate(predictions_result)
    print('Test Accuracy of the Wine Quality Prediction Model:', model_accuracy)

    metrics_evaluator = MulticlassMetrics(evaluation_results.rdd.map(tuple))
    print('Weighted F1 score of the Wine Quality Prediction Model:', metrics_evaluator.weightedFMeasure())
    sys.exit(0)