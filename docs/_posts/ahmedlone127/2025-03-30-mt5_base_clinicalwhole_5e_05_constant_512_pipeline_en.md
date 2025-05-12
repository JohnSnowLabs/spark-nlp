---
layout: model
title: English mt5_base_clinicalwhole_5e_05_constant_512_pipeline pipeline T5Transformer from Mattia2700
author: John Snow Labs
name: mt5_base_clinicalwhole_5e_05_constant_512_pipeline
date: 2025-03-30
tags: [en, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mt5_base_clinicalwhole_5e_05_constant_512_pipeline` is a English model originally trained by Mattia2700.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mt5_base_clinicalwhole_5e_05_constant_512_pipeline_en_5.5.1_3.0_1743300435892.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mt5_base_clinicalwhole_5e_05_constant_512_pipeline_en_5.5.1_3.0_1743300435892.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("mt5_base_clinicalwhole_5e_05_constant_512_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("mt5_base_clinicalwhole_5e_05_constant_512_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mt5_base_clinicalwhole_5e_05_constant_512_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.5 GB|

## References

https://huggingface.co/Mattia2700/mt5-base_ClinicalWhole_5e-05_constant_512

## Included Models

- DocumentAssembler
- T5Transformer