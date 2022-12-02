---
layout: model
title: News Classifier Pipeline for Turkish text
author: John Snow Labs
name: classifierdl_bert_news_pipeline
date: 2021-08-27
tags: [tr, news, classification, open_source]
task: Text Classification
language: tr
edition: Spark NLP 3.2.0
spark_version: 2.4
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pre-trained pipeline classifies Turkish texts of news.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/CLASSIFICATION_TR_NEWS/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/CLASSIFICATION_TR_NEWS.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/classifierdl_bert_news_pipeline_tr_3.2.0_2.4_1630061137177.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("classifierdl_bert_news_pipeline", lang = "tr")

result = pipeline.fullAnnotate("Bonservisi elinde olan Milli oyuncu, yeni takımıyla el sıkıştı")

result["class"]
```
```scala
val pipeline = new PretrainedPipeline("classifierdl_bert_news_pipeline", "tr")

val result = pipeline.fullAnnotate("Bonservisi elinde olan Milli oyuncu, yeni takımıyla el sıkıştı")(0)
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
|Compatibility:|Spark NLP 3.2.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|tr|

## Included Models

- DocumentAssembler
- BertSentenceEmbeddings
- ClassifierDLModel