---
layout: model
title: Esperanto esperberto_malgranda_pipeline pipeline RoBertaEmbeddings from hashk1
author: John Snow Labs
name: esperberto_malgranda_pipeline
date: 2025-03-28
tags: [eo, open_source, pipeline, onnx]
task: Embeddings
language: eo
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`esperberto_malgranda_pipeline` is a Esperanto model originally trained by hashk1.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/esperberto_malgranda_pipeline_eo_5.5.1_3.0_1743127426220.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/esperberto_malgranda_pipeline_eo_5.5.1_3.0_1743127426220.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("esperberto_malgranda_pipeline", lang = "eo")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("esperberto_malgranda_pipeline", lang = "eo")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|esperberto_malgranda_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|eo|
|Size:|308.3 MB|

## References

https://huggingface.co/hashk1/EsperBERTo-malgranda

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings