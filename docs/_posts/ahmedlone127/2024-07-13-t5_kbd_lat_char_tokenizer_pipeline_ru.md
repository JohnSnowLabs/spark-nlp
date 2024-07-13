---
layout: model
title: Russian t5_kbd_lat_char_tokenizer_pipeline pipeline T5Transformer from anzorq
author: John Snow Labs
name: t5_kbd_lat_char_tokenizer_pipeline
date: 2024-07-13
tags: [ru, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: ru
edition: Spark NLP 5.4.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`t5_kbd_lat_char_tokenizer_pipeline` is a Russian model originally trained by anzorq.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/t5_kbd_lat_char_tokenizer_pipeline_ru_5.4.1_3.0_1720888850529.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/t5_kbd_lat_char_tokenizer_pipeline_ru_5.4.1_3.0_1720888850529.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("t5_kbd_lat_char_tokenizer_pipeline", lang = "ru")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("t5_kbd_lat_char_tokenizer_pipeline", lang = "ru")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|t5_kbd_lat_char_tokenizer_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|ru|
|Size:|777.0 MB|

## References

https://huggingface.co/anzorq/kbd_lat-ru_char_tokenizer

## Included Models

- DocumentAssembler
- T5Transformer