---
layout: model
title: Portuguese hatebertimbau_twitter_pipeline pipeline BertForSequenceClassification from knowhate
author: John Snow Labs
name: hatebertimbau_twitter_pipeline
date: 2024-09-14
tags: [pt, open_source, pipeline, onnx]
task: Text Classification
language: pt
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`hatebertimbau_twitter_pipeline` is a Portuguese model originally trained by knowhate.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/hatebertimbau_twitter_pipeline_pt_5.5.0_3.0_1726348004045.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/hatebertimbau_twitter_pipeline_pt_5.5.0_3.0_1726348004045.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("hatebertimbau_twitter_pipeline", lang = "pt")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("hatebertimbau_twitter_pipeline", lang = "pt")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|hatebertimbau_twitter_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|pt|
|Size:|407.8 MB|

## References

https://huggingface.co/knowhate/HateBERTimbau-twitter

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification