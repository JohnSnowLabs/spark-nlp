---
layout: model
title: Japanese comet_t5_base_japanese_pipeline pipeline T5Transformer from nlp-waseda
author: John Snow Labs
name: comet_t5_base_japanese_pipeline
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

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`comet_t5_base_japanese_pipeline` is a Japanese model originally trained by nlp-waseda.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/comet_t5_base_japanese_pipeline_ja_5.4.2_3.0_1722420781187.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/comet_t5_base_japanese_pipeline_ja_5.4.2_3.0_1722420781187.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("comet_t5_base_japanese_pipeline", lang = "ja")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("comet_t5_base_japanese_pipeline", lang = "ja")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|comet_t5_base_japanese_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|ja|
|Size:|948.1 MB|

## References

https://huggingface.co/nlp-waseda/comet-t5-base-japanese

## Included Models

- DocumentAssembler
- T5Transformer