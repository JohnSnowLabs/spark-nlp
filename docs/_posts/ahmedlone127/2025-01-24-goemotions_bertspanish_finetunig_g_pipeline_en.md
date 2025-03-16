---
layout: model
title: English goemotions_bertspanish_finetunig_g_pipeline pipeline BertForSequenceClassification from mrovejaxd
author: John Snow Labs
name: goemotions_bertspanish_finetunig_g_pipeline
date: 2025-01-24
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

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`goemotions_bertspanish_finetunig_g_pipeline` is a English model originally trained by mrovejaxd.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/goemotions_bertspanish_finetunig_g_pipeline_en_5.5.1_3.0_1737710934435.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/goemotions_bertspanish_finetunig_g_pipeline_en_5.5.1_3.0_1737710934435.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("goemotions_bertspanish_finetunig_g_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("goemotions_bertspanish_finetunig_g_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|goemotions_bertspanish_finetunig_g_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|411.8 MB|

## References

https://huggingface.co/mrovejaxd/goemotions_bertspanish_finetunig_g

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification