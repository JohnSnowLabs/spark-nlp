---
layout: model
title: Hebrew hebert_finetuned_precedents_pipeline pipeline BertForQuestionAnswering from shay681
author: John Snow Labs
name: hebert_finetuned_precedents_pipeline
date: 2024-11-11
tags: [he, open_source, pipeline, onnx]
task: Question Answering
language: he
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`hebert_finetuned_precedents_pipeline` is a Hebrew model originally trained by shay681.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/hebert_finetuned_precedents_pipeline_he_5.5.1_3.0_1731307850161.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/hebert_finetuned_precedents_pipeline_he_5.5.1_3.0_1731307850161.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("hebert_finetuned_precedents_pipeline", lang = "he")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("hebert_finetuned_precedents_pipeline", lang = "he")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|hebert_finetuned_precedents_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|he|
|Size:|408.1 MB|

## References

https://huggingface.co/shay681/HeBERT_finetuned_Precedents

## Included Models

- MultiDocumentAssembler
- BertForQuestionAnswering