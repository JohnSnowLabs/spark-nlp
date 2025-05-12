---
layout: model
title: Abkhazian xls_r_abkhaz_test_fitods_pipeline pipeline Wav2Vec2ForCTC from FitoDS
author: John Snow Labs
name: xls_r_abkhaz_test_fitods_pipeline
date: 2025-03-29
tags: [ab, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: ab
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Wav2Vec2ForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`xls_r_abkhaz_test_fitods_pipeline` is a Abkhazian model originally trained by FitoDS.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xls_r_abkhaz_test_fitods_pipeline_ab_5.5.1_3.0_1743279116637.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xls_r_abkhaz_test_fitods_pipeline_ab_5.5.1_3.0_1743279116637.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("xls_r_abkhaz_test_fitods_pipeline", lang = "ab")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("xls_r_abkhaz_test_fitods_pipeline", lang = "ab")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xls_r_abkhaz_test_fitods_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|ab|
|Size:|129.9 KB|

## References

https://huggingface.co/FitoDS/xls-r-ab-test

## Included Models

- AudioAssembler
- Wav2Vec2ForCTC