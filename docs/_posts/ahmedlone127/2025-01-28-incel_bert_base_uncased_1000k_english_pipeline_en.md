---
layout: model
title: English incel_bert_base_uncased_1000k_english_pipeline pipeline BertEmbeddings from pgajo
author: John Snow Labs
name: incel_bert_base_uncased_1000k_english_pipeline
date: 2025-01-28
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

Pretrained BertEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`incel_bert_base_uncased_1000k_english_pipeline` is a English model originally trained by pgajo.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/incel_bert_base_uncased_1000k_english_pipeline_en_5.5.1_3.0_1738035165648.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/incel_bert_base_uncased_1000k_english_pipeline_en_5.5.1_3.0_1738035165648.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("incel_bert_base_uncased_1000k_english_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("incel_bert_base_uncased_1000k_english_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|incel_bert_base_uncased_1000k_english_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|407.1 MB|

## References

https://huggingface.co/pgajo/incel-bert-base-uncased-1000k_english

## Included Models

- DocumentAssembler
- TokenizerModel
- BertEmbeddings