---
layout: model
title: English hw1_part2_ver21_pipeline pipeline BertForQuestionAnswering from weiiiii0622
author: John Snow Labs
name: hw1_part2_ver21_pipeline
date: 2025-04-06
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

Pretrained BertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`hw1_part2_ver21_pipeline` is a English model originally trained by weiiiii0622.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/hw1_part2_ver21_pipeline_en_5.5.1_3.0_1743950727612.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/hw1_part2_ver21_pipeline_en_5.5.1_3.0_1743950727612.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("hw1_part2_ver21_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("hw1_part2_ver21_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|hw1_part2_ver21_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|381.0 MB|

## References

https://huggingface.co/weiiiii0622/HW1_Part2_ver21

## Included Models

- MultiDocumentAssembler
- BertForQuestionAnswering