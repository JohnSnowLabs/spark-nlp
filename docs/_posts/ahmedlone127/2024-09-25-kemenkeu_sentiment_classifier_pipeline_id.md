---
layout: model
title: Indonesian kemenkeu_sentiment_classifier_pipeline pipeline BertForSequenceClassification from hanifnoerr
author: John Snow Labs
name: kemenkeu_sentiment_classifier_pipeline
date: 2024-09-25
tags: [id, open_source, pipeline, onnx]
task: Text Classification
language: id
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`kemenkeu_sentiment_classifier_pipeline` is a Indonesian model originally trained by hanifnoerr.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/kemenkeu_sentiment_classifier_pipeline_id_5.5.0_3.0_1727305152615.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/kemenkeu_sentiment_classifier_pipeline_id_5.5.0_3.0_1727305152615.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("kemenkeu_sentiment_classifier_pipeline", lang = "id")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("kemenkeu_sentiment_classifier_pipeline", lang = "id")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|kemenkeu_sentiment_classifier_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|id|
|Size:|466.4 MB|

## References

https://huggingface.co/hanifnoerr/Kemenkeu-Sentiment-Classifier

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification