---
layout: model
title: English gbert_base_domain_weimar_pipeline pipeline BertEmbeddings from DavidSchweizer
author: John Snow Labs
name: gbert_base_domain_weimar_pipeline
date: 2025-04-09
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

Pretrained BertEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`gbert_base_domain_weimar_pipeline` is a English model originally trained by DavidSchweizer.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/gbert_base_domain_weimar_pipeline_en_5.5.1_3.0_1744190669177.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/gbert_base_domain_weimar_pipeline_en_5.5.1_3.0_1744190669177.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("gbert_base_domain_weimar_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("gbert_base_domain_weimar_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|gbert_base_domain_weimar_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|409.8 MB|

## References

https://huggingface.co/DavidSchweizer/gbert-base-domain-weimar

## Included Models

- DocumentAssembler
- TokenizerModel
- BertEmbeddings