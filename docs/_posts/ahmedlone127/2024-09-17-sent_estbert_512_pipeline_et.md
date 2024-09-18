---
layout: model
title: Estonian sent_estbert_512_pipeline pipeline BertSentenceEmbeddings from tartuNLP
author: John Snow Labs
name: sent_estbert_512_pipeline
date: 2024-09-17
tags: [et, open_source, pipeline, onnx]
task: Embeddings
language: et
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertSentenceEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sent_estbert_512_pipeline` is a Estonian model originally trained by tartuNLP.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_estbert_512_pipeline_et_5.5.0_3.0_1726587872059.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sent_estbert_512_pipeline_et_5.5.0_3.0_1726587872059.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("sent_estbert_512_pipeline", lang = "et")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("sent_estbert_512_pipeline", lang = "et")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sent_estbert_512_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|et|
|Size:|464.0 MB|

## References

https://huggingface.co/tartuNLP/EstBERT_512

## Included Models

- DocumentAssembler
- TokenizerModel
- SentenceDetectorDLModel
- BertSentenceEmbeddings