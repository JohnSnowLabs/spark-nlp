---
layout: model
title: English marian_random_kde4_english_tonga_tonga_islands_french_lisa_z_pipeline pipeline MarianTransformer from Lisa-Z
author: John Snow Labs
name: marian_random_kde4_english_tonga_tonga_islands_french_lisa_z_pipeline
date: 2025-04-06
tags: [en, open_source, pipeline, onnx]
task: Translation
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

Pretrained MarianTransformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`marian_random_kde4_english_tonga_tonga_islands_french_lisa_z_pipeline` is a English model originally trained by Lisa-Z.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/marian_random_kde4_english_tonga_tonga_islands_french_lisa_z_pipeline_en_5.5.1_3.0_1743971354823.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/marian_random_kde4_english_tonga_tonga_islands_french_lisa_z_pipeline_en_5.5.1_3.0_1743971354823.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("marian_random_kde4_english_tonga_tonga_islands_french_lisa_z_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("marian_random_kde4_english_tonga_tonga_islands_french_lisa_z_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|marian_random_kde4_english_tonga_tonga_islands_french_lisa_z_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|396.8 MB|

## References

https://huggingface.co/Lisa-Z/marian-random-kde4-en-to-fr

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- MarianTransformer