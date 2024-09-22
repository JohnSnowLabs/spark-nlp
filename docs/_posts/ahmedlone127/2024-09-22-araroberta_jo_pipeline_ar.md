---
layout: model
title: Arabic araroberta_jo_pipeline pipeline RoBertaEmbeddings from reemalyami
author: John Snow Labs
name: araroberta_jo_pipeline
date: 2024-09-22
tags: [ar, open_source, pipeline, onnx]
task: Embeddings
language: ar
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`araroberta_jo_pipeline` is a Arabic model originally trained by reemalyami.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/araroberta_jo_pipeline_ar_5.5.0_3.0_1726999534008.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/araroberta_jo_pipeline_ar_5.5.0_3.0_1726999534008.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("araroberta_jo_pipeline", lang = "ar")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("araroberta_jo_pipeline", lang = "ar")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|araroberta_jo_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|ar|
|Size:|470.7 MB|

## References

https://huggingface.co/reemalyami/AraRoBERTa-JO

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings