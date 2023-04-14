# FROM python:3.10
# USER root
# RUN mkdir /app
# COPY . /app/
# WORKDIR /app/
# RUN pip install -r requirements.txt
# ENV AIRFLOW_HOME="/app/airflow"
# ENV AIRFLOW_CORE_DAGBAG_IMPORT_TIMEOUT=1000
# ENV AIRFLOW_CORE_ENABLE_XCOM_PICKLING=True
# RUN airflow db init
# RUN airflow users create -e gopalakrishna9101997@gmail.com -f gopala -l krishna -p admin -r Admin -u admin
# RUN chmod 777 start.sh
# RUN apt update -y && apt install awscli -y
# ENTRYPOINT [ "/bin/sh" ]
# CMD [ "start.sh" ]


FROM python:3.10
USER root
RUN mkdir /app
WORKDIR /app/
COPY . /app/

RUN apt update -y && apt install awscli -y

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 unzip -y && pip install -r requirements.txt
CMD ["python3", "main.py"]