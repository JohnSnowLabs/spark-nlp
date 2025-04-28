---
layout: model
title: Hungarian hut5_base_question_generation_pipeline pipeline T5Transformer from edisnord
author: John Snow Labs
name: hut5_base_question_generation_pipeline
date: 2025-03-31
tags: [hu, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: hu
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`hut5_base_question_generation_pipeline` is a Hungarian model originally trained by edisnord.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/hut5_base_question_generation_pipeline_hu_5.5.1_3.0_1743414689962.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/hut5_base_question_generation_pipeline_hu_5.5.1_3.0_1743414689962.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("hut5_base_question_generation_pipeline", lang = "hu")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("hut5_base_question_generation_pipeline", lang = "hu")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|hut5_base_question_generation_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|hu|
|Size:|970.4 MB|

## References

https://huggingface.co/edisnord/hut5-base-question-generation

## Included Models

- DocumentAssembler
- T5Transformer