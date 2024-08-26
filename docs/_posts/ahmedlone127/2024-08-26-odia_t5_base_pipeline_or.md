---
layout: model
title: Oriya (macrolanguage) odia_t5_base_pipeline pipeline T5Transformer from mrSoul7766
author: John Snow Labs
name: odia_t5_base_pipeline
date: 2024-08-26
tags: [or, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: or
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`odia_t5_base_pipeline` is a Oriya (macrolanguage) model originally trained by mrSoul7766.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/odia_t5_base_pipeline_or_5.4.2_3.0_1724685171999.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/odia_t5_base_pipeline_or_5.4.2_3.0_1724685171999.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("odia_t5_base_pipeline", lang = "or")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("odia_t5_base_pipeline", lang = "or")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|odia_t5_base_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|or|
|Size:|1.2 GB|

## References

https://huggingface.co/mrSoul7766/odia-t5-base

## Included Models

- DocumentAssembler
- T5Transformer