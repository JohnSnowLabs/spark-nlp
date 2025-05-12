---
layout: model
title: Swedish sentiment_model_testing_pipeline pipeline DistilBertForSequenceClassification from Alindstroem89
author: John Snow Labs
name: sentiment_model_testing_pipeline
date: 2025-03-28
tags: [sv, open_source, pipeline, onnx]
task: Text Classification
language: sv
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DistilBertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sentiment_model_testing_pipeline` is a Swedish model originally trained by Alindstroem89.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sentiment_model_testing_pipeline_sv_5.5.1_3.0_1743139259800.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sentiment_model_testing_pipeline_sv_5.5.1_3.0_1743139259800.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("sentiment_model_testing_pipeline", lang = "sv")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("sentiment_model_testing_pipeline", lang = "sv")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sentiment_model_testing_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|sv|
|Size:|249.5 MB|

## References

https://huggingface.co/Alindstroem89/sentiment_model_testing

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertForSequenceClassification