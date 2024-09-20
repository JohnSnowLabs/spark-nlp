---
layout: model
title: English sqlcoder_7b_2_arabicsqlv6_pipeline pipeline T5Transformer from ahmedheakl
author: John Snow Labs
name: sqlcoder_7b_2_arabicsqlv6_pipeline
date: 2024-09-12
tags: [en, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: en
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sqlcoder_7b_2_arabicsqlv6_pipeline` is a English model originally trained by ahmedheakl.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sqlcoder_7b_2_arabicsqlv6_pipeline_en_5.5.0_3.0_1726145793170.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sqlcoder_7b_2_arabicsqlv6_pipeline_en_5.5.0_3.0_1726145793170.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("sqlcoder_7b_2_arabicsqlv6_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("sqlcoder_7b_2_arabicsqlv6_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sqlcoder_7b_2_arabicsqlv6_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|179.0 MB|

## References

https://huggingface.co/ahmedheakl/sqlcoder-7b-2-ArabicSQLV6

## Included Models

- DocumentAssembler
- T5Transformer