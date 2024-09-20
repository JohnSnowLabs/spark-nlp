---
layout: model
title: Castilian, Spanish bert_classifier_spanish_news_classification_headlines_untrained_pipeline pipeline BertForSequenceClassification from M47Labs
author: John Snow Labs
name: bert_classifier_spanish_news_classification_headlines_untrained_pipeline
date: 2024-09-04
tags: [es, open_source, pipeline, onnx]
task: Text Classification
language: es
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bert_classifier_spanish_news_classification_headlines_untrained_pipeline` is a Castilian, Spanish model originally trained by M47Labs.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_classifier_spanish_news_classification_headlines_untrained_pipeline_es_5.5.0_3.0_1725432544663.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_classifier_spanish_news_classification_headlines_untrained_pipeline_es_5.5.0_3.0_1725432544663.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bert_classifier_spanish_news_classification_headlines_untrained_pipeline", lang = "es")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bert_classifier_spanish_news_classification_headlines_untrained_pipeline", lang = "es")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_classifier_spanish_news_classification_headlines_untrained_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|es|
|Size:|411.8 MB|

## References

https://huggingface.co/M47Labs/spanish_news_classification_headlines_untrained

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification