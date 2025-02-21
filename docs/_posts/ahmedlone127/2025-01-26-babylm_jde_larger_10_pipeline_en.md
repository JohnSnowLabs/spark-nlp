---
layout: model
title: English babylm_jde_larger_10_pipeline pipeline RoBertaEmbeddings from jdebene
author: John Snow Labs
name: babylm_jde_larger_10_pipeline
date: 2025-01-26
tags: [en, open_source, pipeline, onnx]
task: Embeddings
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

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`babylm_jde_larger_10_pipeline` is a English model originally trained by jdebene.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/babylm_jde_larger_10_pipeline_en_5.5.1_3.0_1737907072660.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/babylm_jde_larger_10_pipeline_en_5.5.1_3.0_1737907072660.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("babylm_jde_larger_10_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("babylm_jde_larger_10_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|babylm_jde_larger_10_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|310.5 MB|

## References

https://huggingface.co/jdebene/BabyLM-jde-larger-10

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings