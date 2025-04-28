---
layout: model
title: Panjabi, Punjabi sent_punjabi_bert_pipeline pipeline BertSentenceEmbeddings from l3cube-pune
author: John Snow Labs
name: sent_punjabi_bert_pipeline
date: 2025-01-31
tags: [pa, open_source, pipeline, onnx]
task: Embeddings
language: pa
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertSentenceEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sent_punjabi_bert_pipeline` is a Panjabi, Punjabi model originally trained by l3cube-pune.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_punjabi_bert_pipeline_pa_5.5.1_3.0_1738359251655.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sent_punjabi_bert_pipeline_pa_5.5.1_3.0_1738359251655.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("sent_punjabi_bert_pipeline", lang = "pa")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("sent_punjabi_bert_pipeline", lang = "pa")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sent_punjabi_bert_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|pa|
|Size:|891.1 MB|

## References

https://huggingface.co/l3cube-pune/punjabi-bert

## Included Models

- DocumentAssembler
- TokenizerModel
- SentenceDetectorDLModel
- BertSentenceEmbeddings