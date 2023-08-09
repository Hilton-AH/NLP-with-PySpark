from fastapi import FastAPI
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, ArrayType
from pyspark.ml.linalg import VectorUDT
from sklearn.metrics.pairwise import cosine_similarity
import uvicorn
import numpy as np
from typing import Union, List, Dict, Any # for Swagger 

app = FastAPI()

WAREHOUSE_PATH = "/opt/warehouse"
# WAREHOUSE_PATH = "./warehouse"

# Define schema
schema = StructType([
    StructField("id", StringType(), True),
    StructField("username", StringType(), True),
    StructField("content", StringType(), True),
    StructField("tokens", ArrayType(StringType()), True),
    StructField("filtered", ArrayType(StringType()), True),
    StructField("tf", VectorUDT(), True),
    StructField("tf_idf", VectorUDT(), True)
])

# Load TF-IDF matrix from Parquet file
# spark = SparkSession.builder \
#     .appName("RestAPI: Load TF-IDF matrix from Parquet") \
#     .getOrCreate()

spark = SparkSession.builder \
        .appName("RestAPI: Load TF-IDF matrix from Parquet") \
        .master("spark://spark-master:7077") \
        .config("spark.executor.instances", 1) \
        .config("spark.cores.max", 2) \
        .config("spark.sql.warehouse.dir", WAREHOUSE_PATH) \
        .getOrCreate()

tfidf_df = spark.read.schema(schema).parquet(WAREHOUSE_PATH)

# http://localhost:8000/api/v1/accounts/
# @app.get("/api/v1/accounts/", response_model=List[Dict[str, Union[str, int]]]) # for Swagger API
@app.get("/api/v1/accounts/")
def get_accounts():
    # Read Parquet file with the defined schema
    tfidf_df = spark.read.schema(schema).parquet(WAREHOUSE_PATH)
    # Convert TF-IDF matrix to Pandas DataFrame for easier manipulation
    tfidf_pd = tfidf_df.toPandas()
    accounts = tfidf_pd[['username', 'id']].to_dict(orient='records')
    return accounts


# http://localhost:8000/api/v1/tf-idf/user-ids/109429260801289174
# @app.get("/api/v1/tf-idf/user-ids/{user_id}", response_model=Dict[str, float]) # for Swagger API
@app.get("/api/v1/tf-idf/user-ids/{user_id}")
def get_tfidf_for_user(user_id: str):
    # Read Parquet file with the defined schema
    tfidf_df = spark.read.schema(schema).parquet(WAREHOUSE_PATH)
    # Convert TF-IDF matrix to Pandas DataFrame for easier manipulation
    tfidf_pd = tfidf_df.toPandas()
    # Get the union of all vocabularies (features)
    all_vocabulary = set()
    for vocab in tfidf_pd['filtered'].values:
        all_vocabulary.update(vocab)
    all_vocabulary = list(all_vocabulary)

    target_row = tfidf_pd.loc[tfidf_pd['id'] == user_id]
    if target_row.empty:
        return {}
    vocabulary = target_row['filtered'].values[0]
    tfidf_values = target_row['tf_idf'].values[0]
    tfidf_dict = dict(zip(vocabulary, tfidf_values))
    # Add zeros for missing words
    for word in all_vocabulary:
        if word not in tfidf_dict:
            tfidf_dict[word] = 0
    return tfidf_dict


# # http://localhost:8000/api/v1/tf-idf/user-ids/109429260801289174/neighbors
# @app.get("/api/v1/tf-idf/user-ids/{user_id}/neighbors", response_model=List[str]) # for Swagger API
@app.get("/api/v1/tf-idf/user-ids/{user_id}/neighbors")
def get_nearest_neighbors(user_id: str, k: int = 10):
    # Read Parquet file with the defined schema
    tfidf_df = spark.read.schema(schema).parquet(WAREHOUSE_PATH)
    # Convert TF-IDF matrix to Pandas DataFrame for easier manipulation
    tfidf_pd = tfidf_df.toPandas()
    # Get the union of all vocabularies (features)
    all_vocabulary = set()
    for vocab in tfidf_pd['filtered'].values:
        all_vocabulary.update(vocab)
    all_vocabulary = list(all_vocabulary)

    # Initialize the matrix with zeros
    num_users = len(tfidf_pd)
    num_features = len(all_vocabulary)
    tf_idf_matrix = np.zeros((num_users, num_features))

    # Fill the matrix with users' TF-IDF vectors
    for index, row in tfidf_pd.iterrows():
        vocab = row['filtered']
        tf_idf_values = row['tf_idf']
        for word, value in zip(vocab, tf_idf_values):
            col_index = all_vocabulary.index(word)
            tf_idf_matrix[index, col_index] = value

    target_row = tfidf_pd.loc[tfidf_pd['id'] == user_id]
    target_index = target_row.index[0]
    target_tfidf_array = tf_idf_matrix[target_index].reshape(1, -1)
    cosine_distances = cosine_similarity(target_tfidf_array, tf_idf_matrix)
    neighbor_indices = np.argsort(cosine_distances.squeeze())[-k:]
    neighbor_ids = tfidf_pd.loc[neighbor_indices, 'id'].values
    return neighbor_ids.tolist()

@app.get("/")
def read_root():
    return {"Hello": "World"}