---
layout: model
title: News Classifier Pipeline for Urdu texts
author: John Snow Labs
name: classifierdl_bert_news_pipeline
date: 2022-02-09
tags: [urdu, news, classifier, pipeline, ur, open_source]
task: Text Classification
language: ur
edition: Spark NLP 3.4.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline classifies Urdu news into up to 7 categories.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/CLASSIFICATION_UR_NEWS/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/CLASSIFICATION_UR_NEWS.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/classifierdl_bert_news_pipeline_ur_3.4.0_3.0_1644402089229.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("classifierdl_bert_news_pipeline", lang = "ur")

result = pipeline.fullAnnotate("""گزشتہ ہفتے ایپل کے حصص میں 11 فیصد اضافہ ہوا ہے۔""")
```
```scala
val pipeline = new PretrainedPipeline("classifierdl_bert_news_pipeline", "ur")

val result = pipeline.fullAnnotate("گزشتہ ہفتے ایپل کے حصص میں 11 فیصد اضافہ ہوا ہے۔")(0)
```
</div>

## Results

```bash
['business']
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|classifierdl_bert_news_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 3.4.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|ur|
|Size:|1.8 GB|

## Included Models

- DocumentAssembler
- BertSentenceEmbeddings
- ClassifierDLModel
