---
layout: model
title: English azbertacontextualizedwordembeddingsinazerbaijanilanguage_pipeline pipeline RoBertaEmbeddings from turalizada
author: John Snow Labs
name: azbertacontextualizedwordembeddingsinazerbaijanilanguage_pipeline
date: 2024-09-20
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

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`azbertacontextualizedwordembeddingsinazerbaijanilanguage_pipeline` is a English model originally trained by turalizada.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/azbertacontextualizedwordembeddingsinazerbaijanilanguage_pipeline_en_5.5.0_3.0_1726857785539.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/azbertacontextualizedwordembeddingsinazerbaijanilanguage_pipeline_en_5.5.0_3.0_1726857785539.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("azbertacontextualizedwordembeddingsinazerbaijanilanguage_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("azbertacontextualizedwordembeddingsinazerbaijanilanguage_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|azbertacontextualizedwordembeddingsinazerbaijanilanguage_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.0 GB|

## References

https://huggingface.co/turalizada/AzBERTaContextualizedWordEmbeddingsinAzerbaijaniLanguage

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings