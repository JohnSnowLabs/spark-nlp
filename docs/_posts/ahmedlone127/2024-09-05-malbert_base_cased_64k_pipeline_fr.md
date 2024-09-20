---
layout: model
title: French malbert_base_cased_64k_pipeline pipeline AlbertEmbeddings from cservan
author: John Snow Labs
name: malbert_base_cased_64k_pipeline
date: 2024-09-05
tags: [fr, open_source, pipeline, onnx]
task: Embeddings
language: fr
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained AlbertEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`malbert_base_cased_64k_pipeline` is a French model originally trained by cservan.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/malbert_base_cased_64k_pipeline_fr_5.5.0_3.0_1725568614463.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/malbert_base_cased_64k_pipeline_fr_5.5.0_3.0_1725568614463.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("malbert_base_cased_64k_pipeline", lang = "fr")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("malbert_base_cased_64k_pipeline", lang = "fr")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|malbert_base_cased_64k_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|fr|
|Size:|37.5 MB|

## References

https://huggingface.co/cservan/malbert-base-cased-64k

## Included Models

- DocumentAssembler
- TokenizerModel
- AlbertEmbeddings