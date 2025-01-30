---
layout: model
title: Arabic bert_labr_unbalanced_pipeline pipeline BertForSequenceClassification from mofawzy
author: John Snow Labs
name: bert_labr_unbalanced_pipeline
date: 2025-01-24
tags: [ar, open_source, pipeline, onnx]
task: Text Classification
language: ar
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bert_labr_unbalanced_pipeline` is a Arabic model originally trained by mofawzy.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_labr_unbalanced_pipeline_ar_5.5.1_3.0_1737710961253.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_labr_unbalanced_pipeline_ar_5.5.1_3.0_1737710961253.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("bert_labr_unbalanced_pipeline", lang = "ar")
annotations =  pipeline.transform(df)
```
```scala
val pipeline = new PretrainedPipeline("bert_labr_unbalanced_pipeline", lang = "ar")
val annotations = pipeline.transform(df)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_labr_unbalanced_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|ar|
|Size:|1.3 GB|

## References

References

https://huggingface.co/mofawzy/bert-labr-unbalanced

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification