---
layout: model
title: Persian mt5_english_persian_farsi_translation_mansoorhamidzadeh_pipeline pipeline T5Transformer from mansoorhamidzadeh
author: John Snow Labs
name: mt5_english_persian_farsi_translation_mansoorhamidzadeh_pipeline
date: 2024-08-03
tags: [fa, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: fa
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mt5_english_persian_farsi_translation_mansoorhamidzadeh_pipeline` is a Persian model originally trained by mansoorhamidzadeh.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mt5_english_persian_farsi_translation_mansoorhamidzadeh_pipeline_fa_5.4.2_3.0_1722700408342.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mt5_english_persian_farsi_translation_mansoorhamidzadeh_pipeline_fa_5.4.2_3.0_1722700408342.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("mt5_english_persian_farsi_translation_mansoorhamidzadeh_pipeline", lang = "fa")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("mt5_english_persian_farsi_translation_mansoorhamidzadeh_pipeline", lang = "fa")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mt5_english_persian_farsi_translation_mansoorhamidzadeh_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|fa|
|Size:|1.2 GB|

## References

https://huggingface.co/mansoorhamidzadeh/mt5_en_fa_translation

## Included Models

- DocumentAssembler
- T5Transformer