---
layout: model
title: English tiny_english_combined_v4_1_0_32_1e_06_cool_sweep_12_pipeline pipeline WhisperForCTC from saahith
author: John Snow Labs
name: tiny_english_combined_v4_1_0_32_1e_06_cool_sweep_12_pipeline
date: 2024-09-21
tags: [en, open_source, pipeline, onnx]
task: Automatic Speech Recognition
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

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`tiny_english_combined_v4_1_0_32_1e_06_cool_sweep_12_pipeline` is a English model originally trained by saahith.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/tiny_english_combined_v4_1_0_32_1e_06_cool_sweep_12_pipeline_en_5.5.0_3.0_1726908317619.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/tiny_english_combined_v4_1_0_32_1e_06_cool_sweep_12_pipeline_en_5.5.0_3.0_1726908317619.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("tiny_english_combined_v4_1_0_32_1e_06_cool_sweep_12_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("tiny_english_combined_v4_1_0_32_1e_06_cool_sweep_12_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|tiny_english_combined_v4_1_0_32_1e_06_cool_sweep_12_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|391.4 MB|

## References

https://huggingface.co/saahith/tiny.en-combined_v4-1-0-32-1e-06-cool-sweep-12

## Included Models

- AudioAssembler
- WhisperForCTC