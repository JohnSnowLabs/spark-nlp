---
layout: model
title: Japanese kogi_mt5_test_pipeline pipeline T5Transformer from kkuramitsu
author: John Snow Labs
name: kogi_mt5_test_pipeline
date: 2024-08-24
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

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`kogi_mt5_test_pipeline` is a Japanese model originally trained by kkuramitsu.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/kogi_mt5_test_pipeline_ja_5.4.2_3.0_1724464498320.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/kogi_mt5_test_pipeline_ja_5.4.2_3.0_1724464498320.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("kogi_mt5_test_pipeline", lang = "ja")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("kogi_mt5_test_pipeline", lang = "ja")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|kogi_mt5_test_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|ja|
|Size:|1.2 GB|

## References

https://huggingface.co/kkuramitsu/kogi-mt5-test

## Included Models

- DocumentAssembler
- T5Transformer