---
layout: model
title: Japanese t5_small_short_pipeline pipeline T5Transformer from retrieva-jp
author: John Snow Labs
name: t5_small_short_pipeline
date: 2024-07-31
tags: [ja, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: ja
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`t5_small_short_pipeline` is a Japanese model originally trained by retrieva-jp.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/t5_small_short_pipeline_ja_5.4.2_3.0_1722419861594.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/t5_small_short_pipeline_ja_5.4.2_3.0_1722419861594.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("t5_small_short_pipeline", lang = "ja")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("t5_small_short_pipeline", lang = "ja")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|t5_small_short_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|ja|
|Size:|349.9 MB|

## References

https://huggingface.co/retrieva-jp/t5-small-short

## Included Models

- DocumentAssembler
- T5Transformer