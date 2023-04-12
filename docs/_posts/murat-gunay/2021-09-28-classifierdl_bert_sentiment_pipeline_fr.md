---
layout: model
title: Sentiment Analysis Pipeline for French texts
author: John Snow Labs
name: classifierdl_bert_sentiment_pipeline
date: 2021-09-28
tags: [fr, sentiment, pipeline, open_source]
task: Sentiment Analysis
language: fr
edition: Spark NLP 3.3.0
spark_version: 2.4
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline identifies the sentiments (positive or negative) in French texts.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/SENTIMENT_FR/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/CLASSIFICATION_Fr_Sentiment.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/classifierdl_bert_sentiment_pipeline_fr_3.3.0_2.4_1632835775093.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/classifierdl_bert_sentiment_pipeline_fr_3.3.0_2.4_1632835775093.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("classifierdl_bert_sentiment_pipeline", lang = "fr")

result = pipeline.annotate("Mignolet vraiment dommage de ne jamais le voir comme titulaire")
```
```scala
val pipeline = new PretrainedPipeline("classifierdl_bert_sentiment_pipeline", lang = "fr")

val result = pipeline.fullAnnotate("Mignolet vraiment dommage de ne jamais le voir comme titulaire")(0)
```
</div>

## Results

```bash
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
|Language:|fr|

## Included Models

- DocumentAssembler
- BertSentenceEmbeddings
- ClassifierDLModel