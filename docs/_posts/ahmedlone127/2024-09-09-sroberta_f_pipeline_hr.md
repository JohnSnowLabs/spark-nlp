---
layout: model
title: Croatian sroberta_f_pipeline pipeline RoBertaEmbeddings from Andrija
author: John Snow Labs
name: sroberta_f_pipeline
date: 2024-09-09
tags: [hr, open_source, pipeline, onnx]
task: Embeddings
language: hr
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sroberta_f_pipeline` is a Croatian model originally trained by Andrija.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sroberta_f_pipeline_hr_5.5.0_3.0_1725883896336.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sroberta_f_pipeline_hr_5.5.0_3.0_1725883896336.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("sroberta_f_pipeline", lang = "hr")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("sroberta_f_pipeline", lang = "hr")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sroberta_f_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|hr|
|Size:|300.3 MB|

## References

https://huggingface.co/Andrija/SRoBERTa-F

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings