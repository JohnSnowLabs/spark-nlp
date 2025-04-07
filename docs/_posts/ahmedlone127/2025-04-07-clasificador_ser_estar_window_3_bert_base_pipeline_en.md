---
layout: model
title: English clasificador_ser_estar_window_3_bert_base_pipeline pipeline BertForSequenceClassification from joheras
author: John Snow Labs
name: clasificador_ser_estar_window_3_bert_base_pipeline
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

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`clasificador_ser_estar_window_3_bert_base_pipeline` is a English model originally trained by joheras.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/clasificador_ser_estar_window_3_bert_base_pipeline_en_5.5.1_3.0_1743993682344.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/clasificador_ser_estar_window_3_bert_base_pipeline_en_5.5.1_3.0_1743993682344.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("clasificador_ser_estar_window_3_bert_base_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("clasificador_ser_estar_window_3_bert_base_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|clasificador_ser_estar_window_3_bert_base_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|627.8 MB|

## References

https://huggingface.co/joheras/clasificador-ser-estar-window-3-bert-base

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification