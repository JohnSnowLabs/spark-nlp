---
layout: model
title: English distillbert_fintuned_with_imdb_dataset_pipeline pipeline DistilBertEmbeddings from jqop
author: John Snow Labs
name: distillbert_fintuned_with_imdb_dataset_pipeline
date: 2025-03-27
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

Pretrained DistilBertEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`distillbert_fintuned_with_imdb_dataset_pipeline` is a English model originally trained by jqop.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distillbert_fintuned_with_imdb_dataset_pipeline_en_5.5.1_3.0_1743108500454.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distillbert_fintuned_with_imdb_dataset_pipeline_en_5.5.1_3.0_1743108500454.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("distillbert_fintuned_with_imdb_dataset_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("distillbert_fintuned_with_imdb_dataset_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distillbert_fintuned_with_imdb_dataset_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|247.2 MB|

## References

https://huggingface.co/jqop/distillBERT-fintuned_with_imdb_dataset

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertEmbeddings