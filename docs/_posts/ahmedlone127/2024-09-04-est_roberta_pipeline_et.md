---
layout: model
title: Estonian est_roberta_pipeline pipeline CamemBertEmbeddings from EMBEDDIA
author: John Snow Labs
name: est_roberta_pipeline
date: 2024-09-04
tags: [et, open_source, pipeline, onnx]
task: Embeddings
language: et
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained CamemBertEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`est_roberta_pipeline` is a Estonian model originally trained by EMBEDDIA.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/est_roberta_pipeline_et_5.5.0_3.0_1725442410460.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/est_roberta_pipeline_et_5.5.0_3.0_1725442410460.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("est_roberta_pipeline", lang = "et")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("est_roberta_pipeline", lang = "et")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|est_roberta_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|et|
|Size:|277.9 MB|

## References

https://huggingface.co/EMBEDDIA/est-roberta

## Included Models

- DocumentAssembler
- TokenizerModel
- CamemBertEmbeddings