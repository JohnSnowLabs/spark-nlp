---
layout: model
title: Dutch, Flemish geobertje_base_dutch_uncased_pipeline pipeline BertEmbeddings from hghcomphys
author: John Snow Labs
name: geobertje_base_dutch_uncased_pipeline
date: 2025-01-28
tags: [nl, open_source, pipeline, onnx]
task: Embeddings
language: nl
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`geobertje_base_dutch_uncased_pipeline` is a Dutch, Flemish model originally trained by hghcomphys.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/geobertje_base_dutch_uncased_pipeline_nl_5.5.1_3.0_1738039598086.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/geobertje_base_dutch_uncased_pipeline_nl_5.5.1_3.0_1738039598086.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("geobertje_base_dutch_uncased_pipeline", lang = "nl")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("geobertje_base_dutch_uncased_pipeline", lang = "nl")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|geobertje_base_dutch_uncased_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|nl|
|Size:|406.4 MB|

## References

https://huggingface.co/hghcomphys/geobertje-base-dutch-uncased

## Included Models

- DocumentAssembler
- TokenizerModel
- BertEmbeddings