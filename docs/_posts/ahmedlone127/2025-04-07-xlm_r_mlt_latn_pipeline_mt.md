---
layout: model
title: Maltese xlm_r_mlt_latn_pipeline pipeline XlmRoBertaEmbeddings from DGurgurov
author: John Snow Labs
name: xlm_r_mlt_latn_pipeline
date: 2025-04-07
tags: [mt, open_source, pipeline, onnx]
task: Embeddings
language: mt
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`xlm_r_mlt_latn_pipeline` is a Maltese model originally trained by DGurgurov.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlm_r_mlt_latn_pipeline_mt_5.5.1_3.0_1744038643060.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xlm_r_mlt_latn_pipeline_mt_5.5.1_3.0_1744038643060.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("xlm_r_mlt_latn_pipeline", lang = "mt")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("xlm_r_mlt_latn_pipeline", lang = "mt")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlm_r_mlt_latn_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|mt|
|Size:|1.0 GB|

## References

https://huggingface.co/DGurgurov/xlm-r_mlt-latn

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaEmbeddings