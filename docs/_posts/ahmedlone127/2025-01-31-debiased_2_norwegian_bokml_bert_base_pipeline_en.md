---
layout: model
title: English debiased_2_norwegian_bokml_bert_base_pipeline pipeline BertEmbeddings from mysil
author: John Snow Labs
name: debiased_2_norwegian_bokml_bert_base_pipeline
date: 2025-01-31
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

Pretrained BertEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`debiased_2_norwegian_bokml_bert_base_pipeline` is a English model originally trained by mysil.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/debiased_2_norwegian_bokml_bert_base_pipeline_en_5.5.1_3.0_1738284745390.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/debiased_2_norwegian_bokml_bert_base_pipeline_en_5.5.1_3.0_1738284745390.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("debiased_2_norwegian_bokml_bert_base_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("debiased_2_norwegian_bokml_bert_base_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|debiased_2_norwegian_bokml_bert_base_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|666.2 MB|

## References

https://huggingface.co/mysil/debiased_2_nb-bert-base

## Included Models

- DocumentAssembler
- TokenizerModel
- BertEmbeddings