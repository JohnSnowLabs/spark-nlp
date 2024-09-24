---
layout: model
title: Danish danish_bert_review_sentiment_pipeline pipeline BertForSequenceClassification from KennethTM
author: John Snow Labs
name: danish_bert_review_sentiment_pipeline
date: 2024-09-22
tags: [da, open_source, pipeline, onnx]
task: Text Classification
language: da
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`danish_bert_review_sentiment_pipeline` is a Danish model originally trained by KennethTM.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/danish_bert_review_sentiment_pipeline_da_5.5.0_3.0_1727029800396.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/danish_bert_review_sentiment_pipeline_da_5.5.0_3.0_1727029800396.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("danish_bert_review_sentiment_pipeline", lang = "da")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("danish_bert_review_sentiment_pipeline", lang = "da")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|danish_bert_review_sentiment_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|da|
|Size:|414.5 MB|

## References

https://huggingface.co/KennethTM/danish-bert-review-sentiment

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification