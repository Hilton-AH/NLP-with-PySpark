FROM apache/spark-py:v3.4.0
USER root
RUN apt-get update && apt-get install -y python3
RUN pip install fastapi uvicorn[standard] pandas pyspark scikit-learn numpy
WORKDIR /opt/app
COPY . /opt/app
RUN chmod a+rx server.sh
ADD mnapi/ mnapi/
ENTRYPOINT [ "/bin/sh", "-c"]
CMD ["./server.sh"]

# Test with:
# docker build -t my-fastapi-app .
# docker run -it --rm -p 8000:8000 my-fastapi-app /bin/bash
# python3 test-api.py