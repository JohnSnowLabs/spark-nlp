---
layout: model
title: English labse_ner_manager_pipeline pipeline BertForTokenClassification from OlegOrwell
author: John Snow Labs
name: labse_ner_manager_pipeline
date: 2025-04-05
tags: [en, open_source, pipeline, onnx]
task: Named Entity Recognition
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

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`labse_ner_manager_pipeline` is a English model originally trained by OlegOrwell.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/labse_ner_manager_pipeline_en_5.5.1_3.0_1743850544246.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/labse_ner_manager_pipeline_en_5.5.1_3.0_1743850544246.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("labse_ner_manager_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("labse_ner_manager_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|labse_ner_manager_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|480.4 MB|

## References

https://huggingface.co/OlegOrwell/LaBSE_ner_manager

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification