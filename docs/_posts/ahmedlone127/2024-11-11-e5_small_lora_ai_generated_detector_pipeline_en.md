---
layout: model
title: English e5_small_lora_ai_generated_detector_pipeline pipeline BertForSequenceClassification from MayZhou
author: John Snow Labs
name: e5_small_lora_ai_generated_detector_pipeline
date: 2024-11-11
tags: [en, open_source, pipeline, onnx]
task: Text Classification
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

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`e5_small_lora_ai_generated_detector_pipeline` is a English model originally trained by MayZhou.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/e5_small_lora_ai_generated_detector_pipeline_en_5.5.1_3.0_1731309394398.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/e5_small_lora_ai_generated_detector_pipeline_en_5.5.1_3.0_1731309394398.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("e5_small_lora_ai_generated_detector_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("e5_small_lora_ai_generated_detector_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|e5_small_lora_ai_generated_detector_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|87.5 MB|

## References

https://huggingface.co/MayZhou/e5-small-lora-ai-generated-detector

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification