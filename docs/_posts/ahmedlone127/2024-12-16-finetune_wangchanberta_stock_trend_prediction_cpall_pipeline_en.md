---
layout: model
title: English finetune_wangchanberta_stock_trend_prediction_cpall_pipeline pipeline CamemBertForSequenceClassification from jab11769
author: John Snow Labs
name: finetune_wangchanberta_stock_trend_prediction_cpall_pipeline
date: 2024-12-16
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

Pretrained CamemBertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`finetune_wangchanberta_stock_trend_prediction_cpall_pipeline` is a English model originally trained by jab11769.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/finetune_wangchanberta_stock_trend_prediction_cpall_pipeline_en_5.5.1_3.0_1734344255150.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/finetune_wangchanberta_stock_trend_prediction_cpall_pipeline_en_5.5.1_3.0_1734344255150.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("finetune_wangchanberta_stock_trend_prediction_cpall_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("finetune_wangchanberta_stock_trend_prediction_cpall_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finetune_wangchanberta_stock_trend_prediction_cpall_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|394.4 MB|

## References

https://huggingface.co/jab11769/Finetune-WangchanBerta-Stock-trend-prediction-CPALL

## Included Models

- DocumentAssembler
- TokenizerModel
- CamemBertForSequenceClassification