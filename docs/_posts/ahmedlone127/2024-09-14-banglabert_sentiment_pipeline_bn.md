---
layout: model
title: Bengali banglabert_sentiment_pipeline pipeline BertForSequenceClassification from ka05ar
author: John Snow Labs
name: banglabert_sentiment_pipeline
date: 2024-09-14
tags: [bn, open_source, pipeline, onnx]
task: Text Classification
language: bn
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`banglabert_sentiment_pipeline` is a Bengali model originally trained by ka05ar.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/banglabert_sentiment_pipeline_bn_5.5.0_3.0_1726348021703.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/banglabert_sentiment_pipeline_bn_5.5.0_3.0_1726348021703.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("banglabert_sentiment_pipeline", lang = "bn")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("banglabert_sentiment_pipeline", lang = "bn")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|banglabert_sentiment_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|bn|
|Size:|414.5 MB|

## References

https://huggingface.co/ka05ar/banglabert-sentiment

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification