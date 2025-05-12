---
layout: model
title: French drbert_umls_kgi_pipeline pipeline CamemBertEmbeddings from a-mannion
author: John Snow Labs
name: drbert_umls_kgi_pipeline
date: 2025-03-27
tags: [fr, open_source, pipeline, onnx]
task: Embeddings
language: fr
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained CamemBertEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`drbert_umls_kgi_pipeline` is a French model originally trained by a-mannion.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/drbert_umls_kgi_pipeline_fr_5.5.1_3.0_1743072789879.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/drbert_umls_kgi_pipeline_fr_5.5.1_3.0_1743072789879.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("drbert_umls_kgi_pipeline", lang = "fr")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("drbert_umls_kgi_pipeline", lang = "fr")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|drbert_umls_kgi_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|fr|
|Size:|412.7 MB|

## References

https://huggingface.co/a-mannion/drbert-umls-kgi

## Included Models

- DocumentAssembler
- TokenizerModel
- CamemBertEmbeddings