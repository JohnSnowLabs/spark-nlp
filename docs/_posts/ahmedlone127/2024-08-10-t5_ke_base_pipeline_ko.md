---
layout: model
title: Korean t5_ke_base_pipeline pipeline T5Transformer from KETI-AIR
author: John Snow Labs
name: t5_ke_base_pipeline
date: 2024-08-10
tags: [ko, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: ko
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`t5_ke_base_pipeline` is a Korean model originally trained by KETI-AIR.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/t5_ke_base_pipeline_ko_5.4.2_3.0_1723332429788.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/t5_ke_base_pipeline_ko_5.4.2_3.0_1723332429788.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("t5_ke_base_pipeline", lang = "ko")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("t5_ke_base_pipeline", lang = "ko")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|t5_ke_base_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|ko|
|Size:|663.2 MB|

## References

https://huggingface.co/KETI-AIR/ke-t5-base-ko

## Included Models

- DocumentAssembler
- T5Transformer