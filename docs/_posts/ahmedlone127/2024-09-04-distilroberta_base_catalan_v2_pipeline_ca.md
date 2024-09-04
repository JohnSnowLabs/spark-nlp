---
layout: model
title: Catalan, Valencian distilroberta_base_catalan_v2_pipeline pipeline RoBertaEmbeddings from projecte-aina
author: John Snow Labs
name: distilroberta_base_catalan_v2_pipeline
date: 2024-09-04
tags: [ca, open_source, pipeline, onnx]
task: Embeddings
language: ca
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`distilroberta_base_catalan_v2_pipeline` is a Catalan, Valencian model originally trained by projecte-aina.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilroberta_base_catalan_v2_pipeline_ca_5.5.0_3.0_1725412529024.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distilroberta_base_catalan_v2_pipeline_ca_5.5.0_3.0_1725412529024.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("distilroberta_base_catalan_v2_pipeline", lang = "ca")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("distilroberta_base_catalan_v2_pipeline", lang = "ca")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilroberta_base_catalan_v2_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|ca|
|Size:|304.1 MB|

## References

https://huggingface.co/projecte-aina/distilroberta-base-ca-v2

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings