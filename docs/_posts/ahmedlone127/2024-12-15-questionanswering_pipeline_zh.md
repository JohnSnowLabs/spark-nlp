---
layout: model
title: Chinese questionanswering_pipeline pipeline BertForQuestionAnswering from QQhahaha
author: John Snow Labs
name: questionanswering_pipeline
date: 2024-12-15
tags: [zh, open_source, pipeline, onnx]
task: Question Answering
language: zh
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`questionanswering_pipeline` is a Chinese model originally trained by QQhahaha.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/questionanswering_pipeline_zh_5.5.1_3.0_1734221953065.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/questionanswering_pipeline_zh_5.5.1_3.0_1734221953065.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("questionanswering_pipeline", lang = "zh")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("questionanswering_pipeline", lang = "zh")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|questionanswering_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|zh|
|Size:|1.2 GB|

## References

https://huggingface.co/QQhahaha/QuestionAnswering

## Included Models

- MultiDocumentAssembler
- BertForQuestionAnswering