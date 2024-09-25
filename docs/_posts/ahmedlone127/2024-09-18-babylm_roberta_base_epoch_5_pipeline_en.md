---
layout: model
title: English babylm_roberta_base_epoch_5_pipeline pipeline RoBertaEmbeddings from Raj-Sanjay-Shah
author: John Snow Labs
name: babylm_roberta_base_epoch_5_pipeline
date: 2024-09-18
tags: [en, open_source, pipeline, onnx]
task: Embeddings
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

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`babylm_roberta_base_epoch_5_pipeline` is a English model originally trained by Raj-Sanjay-Shah.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/babylm_roberta_base_epoch_5_pipeline_en_5.5.0_3.0_1726626629490.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/babylm_roberta_base_epoch_5_pipeline_en_5.5.0_3.0_1726626629490.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("babylm_roberta_base_epoch_5_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("babylm_roberta_base_epoch_5_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|babylm_roberta_base_epoch_5_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|465.6 MB|

## References

https://huggingface.co/Raj-Sanjay-Shah/babyLM_roberta_base_epoch_5

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings