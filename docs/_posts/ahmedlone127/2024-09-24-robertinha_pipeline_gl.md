---
layout: model
title: Galician robertinha_pipeline pipeline RoBertaEmbeddings from mrm8488
author: John Snow Labs
name: robertinha_pipeline
date: 2024-09-24
tags: [gl, open_source, pipeline, onnx]
task: Embeddings
language: gl
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`robertinha_pipeline` is a Galician model originally trained by mrm8488.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/robertinha_pipeline_gl_5.5.0_3.0_1727169273729.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/robertinha_pipeline_gl_5.5.0_3.0_1727169273729.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("robertinha_pipeline", lang = "gl")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("robertinha_pipeline", lang = "gl")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|robertinha_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|gl|
|Size:|311.7 MB|

## References

https://huggingface.co/mrm8488/RoBERTinha

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings