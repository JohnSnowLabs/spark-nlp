---
layout: article
title: Spark NLP Examples
permalink: /docs/en/examples
key: docs-examples
show_edit_on_github: true
show_date: true
modify_date: "2019-05-08"
aside:
  toc: true
sidebar:
  nav: docs-en
header:
  theme: dark
  background: "#123"
---

Showcasing notebooks and codes of how to use Spark NLP in Python and Scala.

## Docker setup

If you want to experience Spark NLP and run Jupyter exmaples without installing anything, you can simply use our [Docker image](https://hub.docker.com/r/johnsnowlabs/spark-nlp-workshop){:target="_blank"}:

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
http://localhost:8888/?token=LOOK_INSIDE_YOUR_CONSOLE
```

## Notebooks

* [Jupyter Notebooks](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/jupyter){:target="_blank"}
* [Strata conference](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/strata){:target="_blank"}
* [Databricks Notebooks](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/databricks){:target="_blank"}
