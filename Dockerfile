FROM centos:7

# Install required dependencies
RUN yum -y update && yum -y install python3 python3-dev python3-pip python3-virtualenv \
    java-1.8.0-openjdk wget

RUN python -V && python3 -V

# Set environment variables
ENV PYSPARK_DRIVER_PYTHON python3
ENV PYSPARK_PYTHON python3

# Install necessary Python packages
RUN pip3 install --upgrade pip && pip3 install numpy pandas pyspark

# Download and configure Apache Spark
RUN wget --no-verbose -O apache-spark.tgz "https://archive.apache.org/dist/spark/spark-3.1.2/spark-3.1.2-bin-hadoop3.2.tgz" \
    && mkdir -p /opt/spark \
    && tar -xf apache-spark.tgz -C /opt/spark --strip-components=1 \
    && rm apache-spark.tgz

RUN ln -s /opt/spark-3.1.2-bin-hadoop2.7 /opt/spark

RUN echo 'export SPARK_HOME=/opt/spark' >> ~/.bashrc \
    && echo 'export PATH=$SPARK_HOME/bin:$PATH' >> ~/.bashrc \
    && echo 'export PYSPARK_PYTHON=python3' >> ~/.bashrc \
    && source ~/.bashrc

# Create directory structure and copy necessary files
RUN mkdir -p /wqpapp9/src /wqpapp9/src/trained_model

COPY src/wqp_single_machine.py /wqpapp9/src/
COPY src/ValidationDataset.csv /wqpapp9/src/
COPY src/trained_model/ /wqpapp9/src/trained_model/

WORKDIR /wqpapp9/src/

ENTRYPOINT ["python3", "wqp_single_machine.py"]