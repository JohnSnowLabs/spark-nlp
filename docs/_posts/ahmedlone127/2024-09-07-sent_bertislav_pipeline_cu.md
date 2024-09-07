---
layout: model
title: Church Slavic, Church Slavonic, Old Bulgarian, Old Church Slavonic, Old Slavonic sent_bertislav_pipeline pipeline BertSentenceEmbeddings from npedrazzini
author: John Snow Labs
name: sent_bertislav_pipeline
date: 2024-09-07
tags: [cu, open_source, pipeline, onnx]
task: Embeddings
language: cu
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertSentenceEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sent_bertislav_pipeline` is a Church Slavic, Church Slavonic, Old Bulgarian, Old Church Slavonic, Old Slavonic model originally trained by npedrazzini.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_bertislav_pipeline_cu_5.5.0_3.0_1725724811280.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sent_bertislav_pipeline_cu_5.5.0_3.0_1725724811280.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("sent_bertislav_pipeline", lang = "cu")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("sent_bertislav_pipeline", lang = "cu")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sent_bertislav_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|cu|
|Size:|667.5 MB|

## References

https://huggingface.co/npedrazzini/BERTislav

## Included Models

- DocumentAssembler
- TokenizerModel
- SentenceDetectorDLModel
- BertSentenceEmbeddings