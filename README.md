# Mastodon Toot Extraction with Natural Language Processing, Apache Spark, Apache Hadoop, PySpark, and MapReduce

In this project I will build a data-pipeline to perform the following steps:

1. Extract recent public "toots" from Mastodon servers and store those into a data-lake.
2. Use Hadoop/Spark to perform a TF-IDF analysis on the "toots".
3. Build a REST service to recommend Mastodon users to follow.

Everything will be running in docker containers on local machine, then deployed on AWS Lambda.

Run `docker-compose up` to bring up entire system and see the Mastodon extractors start running, use the REST API to calculate a TF-IDF matrix, and POST a set of keywords and get a list of recommended Mastodon users.

## Toot Extractor
Create a new docker-compose service called `extractor`. This will run a program that fetches the timeline from one or more Mastodon servers and adds it to "data-lake."

## Component Requirements

* This service should fetch the public-timeline object from one of the mastodon servers -- e.g. <https://mastodon.social/api/v1/timelines/public>, <https://fosstodon.org/api/v1/timelines/public>, etc.
* This service should perform a fetch every 30 seconds and add the new records to the data-lake.
* Store these files in some way that makes it easy to load in aggregate by Spark.

## TF-IDF Matrix Calculator
PySpark application that reads the contents of data-lake, and computes the TF-IDF matrix and stores it in `warehouse` volume -- e.g. as a parquet file.

### Component Requirements

* Program to generate a matrix where each row represents the a Mastodon-user and each column represents a word from our vocabulary.
* A docker-service is implementted that uses pyspark to recompute the TF-IDF every 5 minutes.

## REST API
Exposes this system to analyze relationships between Mastodon users.

### Component Requirements
The RESTful resources include:
* `mstdn-nlp/api/v1/accounts/` -- Lists all of the known matsodon accounts in our data-set.  This should return a list of dictionaries with username and id for each known account -- e.g.
    ```
    [
        {
            "username": "dahlia",
            "id": "109246474478239584"
        },
        {
            "username": "retrohondajunki",
            "id": "109940330642741479"
        },
        ...
    ]
    ```
* `/api/v1/tf-idf/user-ids/<user_id>` -- This should return the TF-IDF matrix row for the given mastodon user as a dictionary with keys = vocabulary words, and values = TF-IDF values.
* `/api/v1/tf-idf/user-ids/<user_id>/neighbors` -- This should return the 10 nearest neighbors, as measured by the cosine-distance between the users's TF-IDF matrix rows.

## Scaling up
In your local environment, you can increase the number of worker containers by running:

```
docker-compose scale spark-worker=6
```
