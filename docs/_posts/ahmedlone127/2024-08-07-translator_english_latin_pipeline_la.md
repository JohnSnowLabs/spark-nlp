---
layout: model
title: Latin translator_english_latin_pipeline pipeline T5Transformer from AlbertY123
author: John Snow Labs
name: translator_english_latin_pipeline
date: 2024-08-07
tags: [la, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: la
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`translator_english_latin_pipeline` is a Latin model originally trained by AlbertY123.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/translator_english_latin_pipeline_la_5.4.2_3.0_1723068262918.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/translator_english_latin_pipeline_la_5.4.2_3.0_1723068262918.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("translator_english_latin_pipeline", lang = "la")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("translator_english_latin_pipeline", lang = "la")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|translator_english_latin_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|la|
|Size:|342.3 MB|

## References

https://huggingface.co/AlbertY123/translator-en-la

## Included Models

- DocumentAssembler
- T5Transformer