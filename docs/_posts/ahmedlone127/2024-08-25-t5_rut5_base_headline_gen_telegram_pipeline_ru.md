---
layout: model
title: Russian t5_rut5_base_headline_gen_telegram_pipeline pipeline T5Transformer from IlyaGusev
author: John Snow Labs
name: t5_rut5_base_headline_gen_telegram_pipeline
date: 2024-08-25
tags: [ru, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: ru
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`t5_rut5_base_headline_gen_telegram_pipeline` is a Russian model originally trained by IlyaGusev.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/t5_rut5_base_headline_gen_telegram_pipeline_ru_5.4.2_3.0_1724628459746.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/t5_rut5_base_headline_gen_telegram_pipeline_ru_5.4.2_3.0_1724628459746.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("t5_rut5_base_headline_gen_telegram_pipeline", lang = "ru")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("t5_rut5_base_headline_gen_telegram_pipeline", lang = "ru")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|t5_rut5_base_headline_gen_telegram_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|ru|
|Size:|995.2 MB|

## References

https://huggingface.co/IlyaGusev/rut5_base_headline_gen_telegram

## Included Models

- DocumentAssembler
- T5Transformer