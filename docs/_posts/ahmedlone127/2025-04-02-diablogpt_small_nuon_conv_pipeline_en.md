---
layout: model
title: English diablogpt_small_nuon_conv_pipeline pipeline GPT2Transformer from heabeoun
author: John Snow Labs
name: diablogpt_small_nuon_conv_pipeline
date: 2025-04-02
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

Pretrained GPT2Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`diablogpt_small_nuon_conv_pipeline` is a English model originally trained by heabeoun.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/diablogpt_small_nuon_conv_pipeline_en_5.5.1_3.0_1743567708353.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/diablogpt_small_nuon_conv_pipeline_en_5.5.1_3.0_1743567708353.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("diablogpt_small_nuon_conv_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("diablogpt_small_nuon_conv_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|diablogpt_small_nuon_conv_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|467.0 MB|

## References

https://huggingface.co/heabeoun/DiabloGPT-small-nuon-conv

## Included Models

- DocumentAssembler
- GPT2Transformer