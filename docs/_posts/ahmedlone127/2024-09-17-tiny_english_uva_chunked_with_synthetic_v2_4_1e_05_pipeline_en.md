---
layout: model
title: English tiny_english_uva_chunked_with_synthetic_v2_4_1e_05_pipeline pipeline WhisperForCTC from saahith
author: John Snow Labs
name: tiny_english_uva_chunked_with_synthetic_v2_4_1e_05_pipeline
date: 2024-09-17
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

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`tiny_english_uva_chunked_with_synthetic_v2_4_1e_05_pipeline` is a English model originally trained by saahith.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/tiny_english_uva_chunked_with_synthetic_v2_4_1e_05_pipeline_en_5.5.0_3.0_1726550825540.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/tiny_english_uva_chunked_with_synthetic_v2_4_1e_05_pipeline_en_5.5.0_3.0_1726550825540.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("tiny_english_uva_chunked_with_synthetic_v2_4_1e_05_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("tiny_english_uva_chunked_with_synthetic_v2_4_1e_05_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|tiny_english_uva_chunked_with_synthetic_v2_4_1e_05_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|394.9 MB|

## References

https://huggingface.co/saahith/tiny.en-uva_chunked_with_synthetic_v2-4-1e-05

## Included Models

- AudioAssembler
- WhisperForCTC