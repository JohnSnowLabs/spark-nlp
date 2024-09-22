---
layout: model
title: English tinybert_general_4l_312d_natureuniverse_pipeline pipeline BertForQuestionAnswering from NatureUniverse
author: John Snow Labs
name: tinybert_general_4l_312d_natureuniverse_pipeline
date: 2024-09-21
tags: [en, open_source, pipeline, onnx]
task: Question Answering
language: en
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`tinybert_general_4l_312d_natureuniverse_pipeline` is a English model originally trained by NatureUniverse.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/tinybert_general_4l_312d_natureuniverse_pipeline_en_5.5.0_3.0_1726928614777.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/tinybert_general_4l_312d_natureuniverse_pipeline_en_5.5.0_3.0_1726928614777.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("tinybert_general_4l_312d_natureuniverse_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("tinybert_general_4l_312d_natureuniverse_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|tinybert_general_4l_312d_natureuniverse_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|53.9 MB|

## References

https://huggingface.co/NatureUniverse/TinyBERT_general_4L_312d

## Included Models

- MultiDocumentAssembler
- BertForQuestionAnswering