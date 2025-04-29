---
layout: model
title: Multilingual bert_mlm_multilingual_cased_pipeline pipeline BertEmbeddings from CLASS-MATE
author: John Snow Labs
name: bert_mlm_multilingual_cased_pipeline
date: 2025-02-07
tags: [xx, open_source, pipeline, onnx]
task: Embeddings
language: xx
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bert_mlm_multilingual_cased_pipeline` is a Multilingual model originally trained by CLASS-MATE.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_mlm_multilingual_cased_pipeline_xx_5.5.1_3.0_1738910908011.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_mlm_multilingual_cased_pipeline_xx_5.5.1_3.0_1738910908011.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bert_mlm_multilingual_cased_pipeline", lang = "xx")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bert_mlm_multilingual_cased_pipeline", lang = "xx")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_mlm_multilingual_cased_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|xx|
|Size:|665.1 MB|

## References

https://huggingface.co/CLASS-MATE/BERT-MLM-multilingual-cased

## Included Models

- DocumentAssembler
- TokenizerModel
- BertEmbeddings