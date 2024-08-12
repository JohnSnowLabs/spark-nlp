---
layout: model
title: Sinhala, Sinhalese t5_mt5_base_sinaha_qa_pipeline pipeline T5Transformer from sankhajay
author: John Snow Labs
name: t5_mt5_base_sinaha_qa_pipeline
date: 2024-07-31
tags: [si, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: si
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`t5_mt5_base_sinaha_qa_pipeline` is a Sinhala, Sinhalese model originally trained by sankhajay.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/t5_mt5_base_sinaha_qa_pipeline_si_5.4.2_3.0_1722416225931.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/t5_mt5_base_sinaha_qa_pipeline_si_5.4.2_3.0_1722416225931.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("t5_mt5_base_sinaha_qa_pipeline", lang = "si")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("t5_mt5_base_sinaha_qa_pipeline", lang = "si")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|t5_mt5_base_sinaha_qa_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|si|
|Size:|1.2 GB|

## References

https://huggingface.co/sankhajay/mt5-base-sinaha-qa

## Included Models

- DocumentAssembler
- T5Transformer