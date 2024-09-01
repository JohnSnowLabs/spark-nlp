---
layout: model
title: Arabic bert_classifier_arabic_marbert_news_article_classification_pipeline pipeline BertForSequenceClassification from Ammar-alhaj-ali
author: John Snow Labs
name: bert_classifier_arabic_marbert_news_article_classification_pipeline
date: 2024-09-01
tags: [ar, open_source, pipeline, onnx]
task: Text Classification
language: ar
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bert_classifier_arabic_marbert_news_article_classification_pipeline` is a Arabic model originally trained by Ammar-alhaj-ali.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_classifier_arabic_marbert_news_article_classification_pipeline_ar_5.4.2_3.0_1725204928842.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_classifier_arabic_marbert_news_article_classification_pipeline_ar_5.4.2_3.0_1725204928842.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bert_classifier_arabic_marbert_news_article_classification_pipeline", lang = "ar")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bert_classifier_arabic_marbert_news_article_classification_pipeline", lang = "ar")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_classifier_arabic_marbert_news_article_classification_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|ar|
|Size:|610.9 MB|

## References

https://huggingface.co/Ammar-alhaj-ali/arabic-MARBERT-news-article-classification

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification