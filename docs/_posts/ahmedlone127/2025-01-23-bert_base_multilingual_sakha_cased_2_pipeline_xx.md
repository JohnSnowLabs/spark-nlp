---
layout: model
title: Multilingual bert_base_multilingual_sakha_cased_2_pipeline pipeline BertEmbeddings from bamaxi
author: John Snow Labs
name: bert_base_multilingual_sakha_cased_2_pipeline
date: 2025-01-23
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

Pretrained BertEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bert_base_multilingual_sakha_cased_2_pipeline` is a Multilingual model originally trained by bamaxi.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_base_multilingual_sakha_cased_2_pipeline_xx_5.5.1_3.0_1737637704620.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_base_multilingual_sakha_cased_2_pipeline_xx_5.5.1_3.0_1737637704620.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bert_base_multilingual_sakha_cased_2_pipeline", lang = "xx")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bert_base_multilingual_sakha_cased_2_pipeline", lang = "xx")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_base_multilingual_sakha_cased_2_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|xx|
|Size:|743.1 MB|

## References

https://huggingface.co/bamaxi/bert-base-multilingual-sakha-cased_2

## Included Models

- DocumentAssembler
- TokenizerModel
- BertEmbeddings