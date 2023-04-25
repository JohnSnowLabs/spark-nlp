# Running Spark NLP in Docker

These example Dockerfiles get get you started with using Spark NLP in a Docker
container.

The following examples set up Jupyter and Scala shells. If you want to run a shell
inside the containers instead, you can specify `bash` at the end of the `docker run`
commands.

## Jupyter Notebook (CPU)

The Dockerfile [SparkNLP-CPU.Dockerfile](SparkNLP-CPU.Dockerfile) sets up a docker
container with Jupyter Notebook. It is based on the official [Jupyter Docker
Images](https://jupyter-docker-stacks.readthedocs.io/en/latest/). To run the notebook on
the default port 8888, we can run

```bash
# Build the Docker Image
docker build -f SparkNLP-CPU.Dockerfile -t sparknlp:latest .

# Run the container and mount the current directory
docker run -it --name sparknlp-container \
    -p 8888:8888 \
    -v "${PWD}":/home/johnsnow/work \
    sparknlp:latest
```

### With GPU Support

If you have compatible NVIDIA GPU, you can use it to leverage better performance on our
machine learning models. Docker provides support for GPU accelerated containers with
[nvidia-docker](https://github.com/NVIDIA/nvidia-docker). The linked repository contains
instructions on how to set it up for your system. (Note that on Windows, using WSL 2
with Docker is
[recommended](https://www.docker.com/blog/wsl-2-gpu-support-for-docker-desktop-on-nvidia-gpus/))

After setting it up, we can use the Dockerfile
[SparkNLP-GPU.Dockerfile](SparkNLP-GPU.Dockerfile) to create an image with CUDA
support. Containers based on this image will then have access to Spark NLP with GPU
acceleration.

The commands to set it up could look like this:

```bash
# Build the image
docker build -f SparkNLP-GPU.Dockerfile -t sparknlp-gpu:latest .

# Start a container with GPU support and mount the current folder
docker run -it --init --name sparknlp-gpu-container \
  -p 8888:8888 \
  -v "${PWD}":/home/johnsnow/work \
  --gpus all \
  --ipc=host \
  sparknlp-gpu:latest
```

*NOTE*: After running the container, don't forget to start Spark NLP with
`sparknlp.start(gpu=True)`! This will set up the right dependencies in Spark.

## Scala Spark Shell

To run Spark NLP in a Scala Spark Shell, we can use the same Dockerfile from Section
[Jupyter Notebook (CPU)](#jupyter-notebook-cpu). However, instead of using the default
entrypoint, we can specify the spark-shell as the command:

```bash
# Run the container, mount the current directory and run spark-shell with Spark NLP
docker run -it --name sparknlp-container \
    -v "${PWD}":/home/johnsnow/work \
    sparknlp:latest \
    /usr/local/spark/bin/spark-shell \
    --conf "spark.driver.memory"="4g" \
    --conf "spark.serializer"="org.apache.spark.serializer.KryoSerializer" \
    --conf "spark.kryoserializer.buffer.max"="2000M" \
    --conf "spark.driver.maxResultSize"="0" \
    --packages "com.johnsnowlabs.nlp:spark-nlp_2.12:4.4.1"
```

To run the shell with GPU support, we use the image from [Jupyter Notebook with GPU
support](#with-gpu-support) and specify the correct package (`spark-nlp-gpu`).

```bash
# Run the container, mount the current directory and run spark-shell with Spark NLP GPU
docker run -it --name sparknlp-container \
    -v "${PWD}":/home/johnsnow/work \
    --gpus all \
    --ipc=host \
    sparknlp-gpu:latest \
    /usr/local/bin/spark-shell \
    --conf "spark.driver.memory"="4g" \
    --conf "spark.serializer"="org.apache.spark.serializer.KryoSerializer" \
    --conf "spark.kryoserializer.buffer.max"="2000M" \
    --conf "spark.driver.maxResultSize"="0" \
    --packages "com.johnsnowlabs.nlp:spark-nlp-gpu_2.12:4.4.1"
```
