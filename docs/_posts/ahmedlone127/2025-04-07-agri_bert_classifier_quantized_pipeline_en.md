---
layout: model
title: English agri_bert_classifier_quantized_pipeline pipeline BertForSequenceClassification from smokxy
author: John Snow Labs
name: agri_bert_classifier_quantized_pipeline
date: 2025-04-07
tags: [en, open_source, pipeline, onnx]
task: Text Classification
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`agri_bert_classifier_quantized_pipeline` is a English model originally trained by smokxy.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/agri_bert_classifier_quantized_pipeline_en_5.5.1_3.0_1744055092726.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/agri_bert_classifier_quantized_pipeline_en_5.5.1_3.0_1744055092726.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("agri_bert_classifier_quantized_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("agri_bert_classifier_quantized_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|agri_bert_classifier_quantized_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|259.9 MB|

## References

https://huggingface.co/smokxy/agri_bert_classifier-quantized

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification