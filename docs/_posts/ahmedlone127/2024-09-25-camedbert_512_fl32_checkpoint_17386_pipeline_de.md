---
layout: model
title: German camedbert_512_fl32_checkpoint_17386_pipeline pipeline BertForTokenClassification from MSey
author: John Snow Labs
name: camedbert_512_fl32_checkpoint_17386_pipeline
date: 2024-09-25
tags: [de, open_source, pipeline, onnx]
task: Named Entity Recognition
language: de
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`camedbert_512_fl32_checkpoint_17386_pipeline` is a German model originally trained by MSey.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/camedbert_512_fl32_checkpoint_17386_pipeline_de_5.5.0_3.0_1727247145941.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/camedbert_512_fl32_checkpoint_17386_pipeline_de_5.5.0_3.0_1727247145941.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("camedbert_512_fl32_checkpoint_17386_pipeline", lang = "de")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("camedbert_512_fl32_checkpoint_17386_pipeline", lang = "de")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|camedbert_512_fl32_checkpoint_17386_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|de|
|Size:|406.9 MB|

## References

https://huggingface.co/MSey/CaMedBERT-512_fl32_checkpoint-17386

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification