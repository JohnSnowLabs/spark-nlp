---
layout: model
title: Multilingual xlmindic_base_uniscript_pipeline pipeline AlbertEmbeddings from ibraheemmoosa
author: John Snow Labs
name: xlmindic_base_uniscript_pipeline
date: 2024-09-04
tags: [xx, open_source, pipeline, onnx]
task: Embeddings
language: xx
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained AlbertEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`xlmindic_base_uniscript_pipeline` is a Multilingual model originally trained by ibraheemmoosa.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlmindic_base_uniscript_pipeline_xx_5.5.0_3.0_1725435287525.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xlmindic_base_uniscript_pipeline_xx_5.5.0_3.0_1725435287525.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("xlmindic_base_uniscript_pipeline", lang = "xx")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("xlmindic_base_uniscript_pipeline", lang = "xx")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlmindic_base_uniscript_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|xx|
|Size:|51.7 MB|

## References

https://huggingface.co/ibraheemmoosa/xlmindic-base-uniscript

## Included Models

- DocumentAssembler
- TokenizerModel
- AlbertEmbeddings