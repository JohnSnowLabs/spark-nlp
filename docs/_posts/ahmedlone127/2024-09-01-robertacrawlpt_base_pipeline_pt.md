---
layout: model
title: Portuguese robertacrawlpt_base_pipeline pipeline RoBertaEmbeddings from eduagarcia
author: John Snow Labs
name: robertacrawlpt_base_pipeline
date: 2024-09-01
tags: [pt, open_source, pipeline, onnx]
task: Embeddings
language: pt
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`robertacrawlpt_base_pipeline` is a Portuguese model originally trained by eduagarcia.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/robertacrawlpt_base_pipeline_pt_5.4.2_3.0_1725191856259.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/robertacrawlpt_base_pipeline_pt_5.4.2_3.0_1725191856259.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("robertacrawlpt_base_pipeline", lang = "pt")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("robertacrawlpt_base_pipeline", lang = "pt")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|robertacrawlpt_base_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|pt|
|Size:|296.9 MB|

## References

https://huggingface.co/eduagarcia/RoBERTaCrawlPT-base

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings