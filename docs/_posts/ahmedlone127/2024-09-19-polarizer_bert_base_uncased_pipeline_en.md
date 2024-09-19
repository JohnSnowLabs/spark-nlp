---
layout: model
title: English polarizer_bert_base_uncased_pipeline pipeline BertEmbeddings from kyungmin011029
author: John Snow Labs
name: polarizer_bert_base_uncased_pipeline
date: 2024-09-19
tags: [en, open_source, pipeline, onnx]
task: Embeddings
language: en
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`polarizer_bert_base_uncased_pipeline` is a English model originally trained by kyungmin011029.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/polarizer_bert_base_uncased_pipeline_en_5.5.0_3.0_1726744775719.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/polarizer_bert_base_uncased_pipeline_en_5.5.0_3.0_1726744775719.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("polarizer_bert_base_uncased_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("polarizer_bert_base_uncased_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|polarizer_bert_base_uncased_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|407.1 MB|

## References

https://huggingface.co/kyungmin011029/Polarizer-bert-base-uncased

## Included Models

- DocumentAssembler
- TokenizerModel
- BertEmbeddings