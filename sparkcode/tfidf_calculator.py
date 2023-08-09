from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF
from pyspark.sql.functions import lower, regexp_replace, col
from pyspark.sql.types import StructType, StructField, StringType, Row

import os
import time
import json

DATA_LAKE_PATH = "/opt/datalake/"
WAREHOUSE_PATH = "/opt/warehouse/"

def main():
    spark = SparkSession.builder \
        .appName("TF-IDF Calculator") \
        .master("spark://spark-master:7077") \
        .config("spark.executor.instances", 1) \
        .config("spark.cores.max", 2) \
        .config("spark.sql.warehouse.dir", WAREHOUSE_PATH) \
        .getOrCreate()
    # spark = SparkSession.builder \
    #     .appName("TF-IDF Calculator") \
    #     .getOrCreate()

    while True:
        files = os.listdir(DATA_LAKE_PATH)
        for file in files:
            file_path = os.path.join(DATA_LAKE_PATH, file)

            # Define schema
            schema = StructType([
                StructField("account", StructType(
                    StructType([
                        StructField("id", StringType(), True),
                        StructField("username", StringType(), True)
                    ])
                ), True),
                StructField("content", StringType(), True)
            ])

            # Create an empty DataFrame with the defined schema
            df = spark.createDataFrame([], schema)

            # Read the JSON file as a text file
            with open(file_path, "r", encoding="utf-8-sig") as file:
                file_content = file.read()
            
            # Handle bad JSON files
            try:
                json_array = json.loads(file_content)
            except json.JSONDecodeError as e:
                print(f"Error parsing file {file_path}: {e}")
                print(f"File content: {file_content}")
                os.remove(file_path)
                continue

            # Extract only the fields defined in the schema for each JSON object in the array
            filtered_objects = [
                {field.name: json_object[field.name] for field in schema.fields}
                for json_object in json_array
            ]

            # Convert the filtered JSON objects to PySpark Rows
            rows = [Row(**filtered_object) for filtered_object in filtered_objects]

            # Create a DataFrame with the rows and schema
            df = spark.createDataFrame(rows, schema)

            # Extract the id and username fields from the account field
            df = df.select(
                col("account.id").alias("id"),
                col("account.username").alias("username"),
                "content"
            )

            # Show the DataFrame
            # df.show(truncate=False)

            # Preprocess data
            df = df.select("id", "username", lower(col("content")).alias("content"))
            df = df.withColumn("content", regexp_replace("content", "[^a-zA-Z0-9\\s,.!?]", ""))
            tokenizer = Tokenizer(inputCol="content", outputCol="tokens")
            df = tokenizer.transform(df)
            remover = StopWordsRemover(inputCol="tokens", outputCol="filtered")
            df = remover.transform(df)

            # Compute term frequency and inverse document frequency
            cv = CountVectorizer(inputCol="filtered", outputCol="tf")
            cv_model = cv.fit(df)
            df = cv_model.transform(df)
            idf = IDF(inputCol="tf", outputCol="tf_idf")
            idf_model = idf.fit(df)
            df = idf_model.transform(df)

            # Save TF-IDF matrix to Parquet file
            df.write.mode("append").parquet(WAREHOUSE_PATH)

            df.show(2)

            # Remove processed file
            os.remove(file_path)

        # break
        time.sleep(300)

    spark.stop()

if __name__ == "__main__":
    main()