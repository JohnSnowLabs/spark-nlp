---
layout: model
title: Slovenian sloberta_finetuned_dlib_1850_1919_pipeline pipeline CamemBertEmbeddings from janezb
author: John Snow Labs
name: sloberta_finetuned_dlib_1850_1919_pipeline
date: 2024-09-09
tags: [sl, open_source, pipeline, onnx]
task: Embeddings
language: sl
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained CamemBertEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sloberta_finetuned_dlib_1850_1919_pipeline` is a Slovenian model originally trained by janezb.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sloberta_finetuned_dlib_1850_1919_pipeline_sl_5.5.0_3.0_1725851418626.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sloberta_finetuned_dlib_1850_1919_pipeline_sl_5.5.0_3.0_1725851418626.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("sloberta_finetuned_dlib_1850_1919_pipeline", lang = "sl")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("sloberta_finetuned_dlib_1850_1919_pipeline", lang = "sl")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sloberta_finetuned_dlib_1850_1919_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|sl|
|Size:|410.5 MB|

## References

https://huggingface.co/janezb/sloberta-finetuned-dlib-1850-1919

## Included Models

- DocumentAssembler
- TokenizerModel
- CamemBertEmbeddings