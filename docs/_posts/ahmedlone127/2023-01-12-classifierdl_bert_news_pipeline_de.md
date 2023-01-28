---
layout: model
title: News Classifier Pipeline for German text
author: John Snow Labs
name: classifierdl_bert_news_pipeline
date: 2023-01-12
tags: [de, classifier, pipeline, news, open_source]
task: Text Classification
language: de
edition: Spark NLP 4.3.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pre-trained pipeline classifies German texts of news.

## Predicted Entities



{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/CLASSIFICATION_DE_NEWS/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/CLASSIFICATION_DE_NEWS.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/classifierdl_bert_news_pipeline_de_4.3.0_3.0_1673541411059.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/classifierdl_bert_news_pipeline_de_4.3.0_3.0_1673541411059.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("classifierdl_bert_news_pipeline", lang = "de")

result = pipeline.fullAnnotate("""Niki Lauda in einem McLaren MP 4/2 TAG Turbo. Mit diesem Gefährt sicherte sich der Österreicher 1984 seinen dritten Weltmeistertitel, einen halben (!)""")
```
```scala

val pipeline = new PretrainedPipeline("classifierdl_bert_news_pipeline", "de")

val result = pipeline.fullAnnotate("Niki Lauda in einem McLaren MP 4/2 TAG Turbo. Mit diesem Gefährt sicherte sich der Österreicher 1984 seinen dritten Weltmeistertitel, einen halben (!)")(0)
```
</div>

## Results

```bash

["Sport"]
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|classifierdl_bert_news_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.3.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|de|
|Size:|693.1 MB|

## Included Models

- DocumentAssembler
- BertSentenceEmbeddings
- ClassifierDLModel