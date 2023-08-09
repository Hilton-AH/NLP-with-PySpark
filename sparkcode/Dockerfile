FROM apache/spark-py:v3.4.0
USER root
RUN apt-get update && apt-get install -y python3
RUN pip install numpy pyspark
RUN ln -s /usr/bin/python3 /usr/bin/python
WORKDIR /opt/app
COPY tfidf_calculator.py ./sparkcode/
# COPY ./data/*.json /opt/datalake/
CMD ["python", "./sparkcode/tfidf_calculator.py"]

# Test with:
# docker build -t my-spark-image . 
# docker run -it my-spark-image /bin/bash
# python ./sparkcode/tfidf_calculator.py