---
layout: model
title: Russian rubert_base_cased_nli_threeway_pipeline pipeline BertForZeroShotClassification from cointegrated
author: John Snow Labs
name: rubert_base_cased_nli_threeway_pipeline
date: 2025-05-27
tags: [ru, open_source, pipeline, onnx]
task: Zero-Shot Classification
language: ru
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForZeroShotClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`rubert_base_cased_nli_threeway_pipeline` is a Russian model originally trained by cointegrated.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/rubert_base_cased_nli_threeway_pipeline_ru_5.5.1_3.0_1748372000481.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/rubert_base_cased_nli_threeway_pipeline_ru_5.5.1_3.0_1748372000481.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("rubert_base_cased_nli_threeway_pipeline", lang = "ru")
annotations =  pipeline.transform(df)
```
```scala
val pipeline = new PretrainedPipeline("rubert_base_cased_nli_threeway_pipeline", lang = "ru")
val annotations = pipeline.transform(df)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|rubert_base_cased_nli_threeway_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|ru|
|Size:|664.4 MB|

## References

References

https://huggingface.co/cointegrated/rubert-base-cased-nli-threeway

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForZeroShotClassification