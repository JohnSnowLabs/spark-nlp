---
layout: model
title: Italian cross_encoder_bert_base_stsb_pipeline pipeline BertForSequenceClassification from efederici
author: John Snow Labs
name: cross_encoder_bert_base_stsb_pipeline
date: 2024-09-26
tags: [it, open_source, pipeline, onnx]
task: Text Classification
language: it
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`cross_encoder_bert_base_stsb_pipeline` is a Italian model originally trained by efederici.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/cross_encoder_bert_base_stsb_pipeline_it_5.5.0_3.0_1727343000979.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/cross_encoder_bert_base_stsb_pipeline_it_5.5.0_3.0_1727343000979.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("cross_encoder_bert_base_stsb_pipeline", lang = "it")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("cross_encoder_bert_base_stsb_pipeline", lang = "it")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|cross_encoder_bert_base_stsb_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|it|
|Size:|414.8 MB|

## References

https://huggingface.co/efederici/cross-encoder-bert-base-stsb

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification