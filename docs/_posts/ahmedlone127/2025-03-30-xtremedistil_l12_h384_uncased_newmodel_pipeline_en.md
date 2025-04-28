---
layout: model
title: English xtremedistil_l12_h384_uncased_newmodel_pipeline pipeline BertForQuestionAnswering from tachyon-11
author: John Snow Labs
name: xtremedistil_l12_h384_uncased_newmodel_pipeline
date: 2025-03-30
tags: [en, open_source, pipeline, onnx]
task: Question Answering
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

Pretrained BertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`xtremedistil_l12_h384_uncased_newmodel_pipeline` is a English model originally trained by tachyon-11.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xtremedistil_l12_h384_uncased_newmodel_pipeline_en_5.5.1_3.0_1743345057984.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xtremedistil_l12_h384_uncased_newmodel_pipeline_en_5.5.1_3.0_1743345057984.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("xtremedistil_l12_h384_uncased_newmodel_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("xtremedistil_l12_h384_uncased_newmodel_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xtremedistil_l12_h384_uncased_newmodel_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|124.0 MB|

## References

https://huggingface.co/tachyon-11/xtremedistil-l12-h384-uncased-newmodel

## Included Models

- MultiDocumentAssembler
- BertForQuestionAnswering