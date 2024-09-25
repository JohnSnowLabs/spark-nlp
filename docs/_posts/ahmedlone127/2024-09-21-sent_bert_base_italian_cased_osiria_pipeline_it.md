---
layout: model
title: Italian sent_bert_base_italian_cased_osiria_pipeline pipeline BertSentenceEmbeddings from osiria
author: John Snow Labs
name: sent_bert_base_italian_cased_osiria_pipeline
date: 2024-09-21
tags: [it, open_source, pipeline, onnx]
task: Embeddings
language: it
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertSentenceEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sent_bert_base_italian_cased_osiria_pipeline` is a Italian model originally trained by osiria.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_bert_base_italian_cased_osiria_pipeline_it_5.5.0_3.0_1726898316037.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sent_bert_base_italian_cased_osiria_pipeline_it_5.5.0_3.0_1726898316037.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("sent_bert_base_italian_cased_osiria_pipeline", lang = "it")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("sent_bert_base_italian_cased_osiria_pipeline", lang = "it")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sent_bert_base_italian_cased_osiria_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|it|
|Size:|409.5 MB|

## References

https://huggingface.co/osiria/bert-base-italian-cased

## Included Models

- DocumentAssembler
- TokenizerModel
- SentenceDetectorDLModel
- BertSentenceEmbeddings