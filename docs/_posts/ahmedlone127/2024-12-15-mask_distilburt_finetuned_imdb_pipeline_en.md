---
layout: model
title: English mask_distilburt_finetuned_imdb_pipeline pipeline DistilBertEmbeddings from Faizyhugging
author: John Snow Labs
name: mask_distilburt_finetuned_imdb_pipeline
date: 2024-12-15
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

Pretrained DistilBertEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mask_distilburt_finetuned_imdb_pipeline` is a English model originally trained by Faizyhugging.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mask_distilburt_finetuned_imdb_pipeline_en_5.5.1_3.0_1734289931323.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mask_distilburt_finetuned_imdb_pipeline_en_5.5.1_3.0_1734289931323.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("mask_distilburt_finetuned_imdb_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("mask_distilburt_finetuned_imdb_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mask_distilburt_finetuned_imdb_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|247.2 MB|

## References

https://huggingface.co/Faizyhugging/Mask-distilburt-finetuned-imdb

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertEmbeddings