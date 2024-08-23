---
layout: model
title: English kltn_coqe_vit5_total_pasol_v4_pipeline pipeline T5Transformer from ThuyNT03
author: John Snow Labs
name: kltn_coqe_vit5_total_pasol_v4_pipeline
date: 2024-08-23
tags: [en, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: en
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`kltn_coqe_vit5_total_pasol_v4_pipeline` is a English model originally trained by ThuyNT03.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/kltn_coqe_vit5_total_pasol_v4_pipeline_en_5.4.2_3.0_1724451113989.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/kltn_coqe_vit5_total_pasol_v4_pipeline_en_5.4.2_3.0_1724451113989.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("kltn_coqe_vit5_total_pasol_v4_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("kltn_coqe_vit5_total_pasol_v4_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|kltn_coqe_vit5_total_pasol_v4_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|3.0 GB|

## References

https://huggingface.co/ThuyNT03/KLTN_COQE_viT5_total_PASOL_v4

## Included Models

- DocumentAssembler
- T5Transformer