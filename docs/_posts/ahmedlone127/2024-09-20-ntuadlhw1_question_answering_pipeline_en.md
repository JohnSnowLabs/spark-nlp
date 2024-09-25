---
layout: model
title: English ntuadlhw1_question_answering_pipeline pipeline BertForQuestionAnswering from weitung8
author: John Snow Labs
name: ntuadlhw1_question_answering_pipeline
date: 2024-09-20
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

Pretrained BertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`ntuadlhw1_question_answering_pipeline` is a English model originally trained by weitung8.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ntuadlhw1_question_answering_pipeline_en_5.5.0_3.0_1726834427974.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/ntuadlhw1_question_answering_pipeline_en_5.5.0_3.0_1726834427974.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("ntuadlhw1_question_answering_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("ntuadlhw1_question_answering_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ntuadlhw1_question_answering_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.2 GB|

## References

https://huggingface.co/weitung8/ntuadlhw1-question-answering

## Included Models

- MultiDocumentAssembler
- BertForQuestionAnswering