FROM python:3.9

WORKDIR /opt/app
COPY extractor.py ./extractor/
RUN chmod +x ./extractor/extractor.py
RUN pip install requests
CMD ["python", "./extractor/extractor.py"]

# Test with:
# docker build -t my-extractor-image . 
# docker run -it my-extractor-image /bin/bash