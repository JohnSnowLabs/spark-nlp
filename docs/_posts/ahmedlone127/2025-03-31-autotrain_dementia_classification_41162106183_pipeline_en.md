---
layout: model
title: English autotrain_dementia_classification_41162106183_pipeline pipeline ViTForImageClassification from RiniPL
author: John Snow Labs
name: autotrain_dementia_classification_41162106183_pipeline
date: 2025-03-31
tags: [en, open_source, pipeline, onnx]
task: Image Classification
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

Pretrained ViTForImageClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`autotrain_dementia_classification_41162106183_pipeline` is a English model originally trained by RiniPL.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/autotrain_dementia_classification_41162106183_pipeline_en_5.5.1_3.0_1743444938678.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/autotrain_dementia_classification_41162106183_pipeline_en_5.5.1_3.0_1743444938678.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("autotrain_dementia_classification_41162106183_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("autotrain_dementia_classification_41162106183_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|autotrain_dementia_classification_41162106183_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|321.2 MB|

## References

https://huggingface.co/RiniPL/autotrain-dementia_classification-41162106183

## Included Models

- ImageAssembler
- ViTForImageClassification