---
layout: model
title: English debugg_datasetqas_debugdataset_with_debugmodel_pipeline pipeline BertForQuestionAnswering from muhammadravi251001
author: John Snow Labs
name: debugg_datasetqas_debugdataset_with_debugmodel_pipeline
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

Pretrained BertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`debugg_datasetqas_debugdataset_with_debugmodel_pipeline` is a English model originally trained by muhammadravi251001.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/debugg_datasetqas_debugdataset_with_debugmodel_pipeline_en_5.5.1_3.0_1743345066956.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/debugg_datasetqas_debugdataset_with_debugmodel_pipeline_en_5.5.1_3.0_1743345066956.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("debugg_datasetqas_debugdataset_with_debugmodel_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("debugg_datasetqas_debugdataset_with_debugmodel_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|debugg_datasetqas_debugdataset_with_debugmodel_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|411.7 MB|

## References

https://huggingface.co/muhammadravi251001/DEBUGG-DatasetQAS-DEBUGdataset-with-DEBUGmodel

## Included Models

- MultiDocumentAssembler
- BertForQuestionAnswering