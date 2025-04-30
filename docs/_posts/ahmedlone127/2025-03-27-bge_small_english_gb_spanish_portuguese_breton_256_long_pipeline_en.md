---
layout: model
title: English bge_small_english_gb_spanish_portuguese_breton_256_long_pipeline pipeline BGEEmbeddings from jvanhoof
author: John Snow Labs
name: bge_small_english_gb_spanish_portuguese_breton_256_long_pipeline
date: 2025-03-27
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

Pretrained BGEEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bge_small_english_gb_spanish_portuguese_breton_256_long_pipeline` is a English model originally trained by jvanhoof.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bge_small_english_gb_spanish_portuguese_breton_256_long_pipeline_en_5.5.1_3.0_1743074804080.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bge_small_english_gb_spanish_portuguese_breton_256_long_pipeline_en_5.5.1_3.0_1743074804080.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bge_small_english_gb_spanish_portuguese_breton_256_long_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bge_small_english_gb_spanish_portuguese_breton_256_long_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bge_small_english_gb_spanish_portuguese_breton_256_long_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|123.8 MB|

## References

https://huggingface.co/jvanhoof/bge-small-en-gb-es-pt-br-256-long

## Included Models

- DocumentAssembler
- BGEEmbeddings