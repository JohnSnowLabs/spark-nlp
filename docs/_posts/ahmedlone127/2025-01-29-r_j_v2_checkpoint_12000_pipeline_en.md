---
layout: model
title: English r_j_v2_checkpoint_12000_pipeline pipeline RoBertaEmbeddings from datalawyer
author: John Snow Labs
name: r_j_v2_checkpoint_12000_pipeline
date: 2025-01-29
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

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`r_j_v2_checkpoint_12000_pipeline` is a English model originally trained by datalawyer.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/r_j_v2_checkpoint_12000_pipeline_en_5.5.1_3.0_1738187501427.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/r_j_v2_checkpoint_12000_pipeline_en_5.5.1_3.0_1738187501427.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("r_j_v2_checkpoint_12000_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("r_j_v2_checkpoint_12000_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|r_j_v2_checkpoint_12000_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|466.2 MB|

## References

https://huggingface.co/datalawyer/r_j_v2_checkpoint_12000

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings