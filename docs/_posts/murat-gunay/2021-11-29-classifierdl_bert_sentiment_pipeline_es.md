---
layout: model
title: Sentiment Analysis Pipeline for Spanish texts
author: John Snow Labs
name: classifierdl_bert_sentiment_pipeline
date: 2021-11-29
tags: [spanish, sentiment, es, classifier, open_source]
task: Sentiment Analysis
language: es
edition: Spark NLP 3.3.0
spark_version: 2.4
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline identifies the sentiments (positive or negative) in Spanish texts.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/CLASSIFICATION_ES_SENTIMENT.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/classifierdl_bert_sentiment_pipeline_es_3.3.0_2.4_1638192149292.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("classifierdl_bert_sentiment_pipeline", lang = "es")

result1 = pipeline.annotate("Estoy seguro de que esta vez pasará la entrevista.")

result2 = pipeline.annotate("Soy una persona que intenta desayunar todas las mañanas sin falta.")

result3 = pipeline.annotate("No estoy seguro de si mi salario mensual es suficiente para vivir.")
```
```scala
val pipeline = new PretrainedPipeline("classifierdl_bert_sentiment_pipeline", lang = "es")

val result1 = pipeline.annotate("Estoy seguro de que esta vez pasará la entrevista.")(0)

val result2 = pipeline.annotate("Soy una persona que intenta desayunar todas las mañanas sin falta.")(0)

val result3 = pipeline.annotate("No estoy seguro de si mi salario mensual es suficiente para vivir.")(0)
```
</div>

## Results

```bash
['POSITIVE']
['NEUTRAL']
['NEGATIVE']
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|classifierdl_bert_sentiment_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 3.3.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|es|

## Included Models

- DocumentAssembler
- BertSentenceEmbeddings
- ClassifierDLModel