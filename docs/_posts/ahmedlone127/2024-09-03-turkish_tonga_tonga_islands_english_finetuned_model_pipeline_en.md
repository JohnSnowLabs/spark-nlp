---
layout: model
title: English turkish_tonga_tonga_islands_english_finetuned_model_pipeline pipeline MarianTransformer from ckartal
author: John Snow Labs
name: turkish_tonga_tonga_islands_english_finetuned_model_pipeline
date: 2024-09-03
tags: [en, open_source, pipeline, onnx]
task: Translation
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

Pretrained MarianTransformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`turkish_tonga_tonga_islands_english_finetuned_model_pipeline` is a English model originally trained by ckartal.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/turkish_tonga_tonga_islands_english_finetuned_model_pipeline_en_5.5.0_3.0_1725346880724.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/turkish_tonga_tonga_islands_english_finetuned_model_pipeline_en_5.5.0_3.0_1725346880724.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("turkish_tonga_tonga_islands_english_finetuned_model_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("turkish_tonga_tonga_islands_english_finetuned_model_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|turkish_tonga_tonga_islands_english_finetuned_model_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|525.8 MB|

## References

https://huggingface.co/ckartal/turkish-to-english-finetuned-model

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- MarianTransformer