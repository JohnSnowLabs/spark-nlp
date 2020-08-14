---
layout: article
title: Examples
key: docs-examples
permalink: /docs/en/examples
modify_date: "2019-05-16"
---

Showcasing notebooks and codes of how to use Spark NLP in Python and Scala.

## Docker setup


If you want to experience Spark NLP and run Jupyter examples without installing anything, you can simply use our [Docker image](https://hub.docker.com/r/johnsnowlabs/spark-nlp-workshop):

1- Get the docker image for spark-nlp-workshop:

```bash
docker pull johnsnowlabs/spark-nlp-workshop
```

2- Run the image locally with port binding.

```bash
 docker run -it --rm -p 8888:8888 -p 4040:4040 johnsnowlabs/spark-nlp-workshop
```

3- Open Jupyter notebooks inside your browser by using the token printed on the console.

```bash
http://localhost:8888/
```

* The password to Jupyter notebook is `sparknlp`
* The size of the image grows everytime you download a pretrained model or a pretrained pipeline. You can cleanup `~/cache_pretrained` if you don't need them.
* This docker image is only meant for testing/learning purposes and should not be used in production environments. Please install Spark NLP natively.
* There are lots of notebooks for Google Colab. If you intend to use those inside the Docker you should skip the Java intallation part

## Notebooks

* [Tutorials and trainings](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials)
* [Jupyter Notebooks](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/jupyter)
* [Databricks Notebooks](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/databricks)
