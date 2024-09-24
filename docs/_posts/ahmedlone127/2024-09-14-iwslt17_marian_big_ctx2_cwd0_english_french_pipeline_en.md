---
layout: model
title: English iwslt17_marian_big_ctx2_cwd0_english_french_pipeline pipeline MarianTransformer from context-mt
author: John Snow Labs
name: iwslt17_marian_big_ctx2_cwd0_english_french_pipeline
date: 2024-09-14
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

Pretrained MarianTransformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`iwslt17_marian_big_ctx2_cwd0_english_french_pipeline` is a English model originally trained by context-mt.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/iwslt17_marian_big_ctx2_cwd0_english_french_pipeline_en_5.5.0_3.0_1726350872257.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/iwslt17_marian_big_ctx2_cwd0_english_french_pipeline_en_5.5.0_3.0_1726350872257.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("iwslt17_marian_big_ctx2_cwd0_english_french_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("iwslt17_marian_big_ctx2_cwd0_english_french_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|iwslt17_marian_big_ctx2_cwd0_english_french_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.3 GB|

## References

https://huggingface.co/context-mt/iwslt17-marian-big-ctx2-cwd0-en-fr

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- MarianTransformer