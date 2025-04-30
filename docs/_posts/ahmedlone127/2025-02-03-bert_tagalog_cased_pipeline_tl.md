---
layout: model
title: Tagalog bert_tagalog_cased_pipeline pipeline BertEmbeddings from dost-asti
author: John Snow Labs
name: bert_tagalog_cased_pipeline
date: 2025-02-03
tags: [tl, open_source, pipeline, onnx]
task: Embeddings
language: tl
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bert_tagalog_cased_pipeline` is a Tagalog model originally trained by dost-asti.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_tagalog_cased_pipeline_tl_5.5.1_3.0_1738549013004.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_tagalog_cased_pipeline_tl_5.5.1_3.0_1738549013004.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bert_tagalog_cased_pipeline", lang = "tl")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bert_tagalog_cased_pipeline", lang = "tl")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_tagalog_cased_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|tl|
|Size:|406.9 MB|

## References

https://huggingface.co/dost-asti/BERT-tl-cased

## Included Models

- DocumentAssembler
- TokenizerModel
- BertEmbeddings