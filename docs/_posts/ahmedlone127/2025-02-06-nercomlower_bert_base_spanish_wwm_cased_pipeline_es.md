---
layout: model
title: Castilian, Spanish nercomlower_bert_base_spanish_wwm_cased_pipeline pipeline BertForTokenClassification from simonestradasch
author: John Snow Labs
name: nercomlower_bert_base_spanish_wwm_cased_pipeline
date: 2025-02-06
tags: [es, open_source, pipeline, onnx]
task: Named Entity Recognition
language: es
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`nercomlower_bert_base_spanish_wwm_cased_pipeline` is a Castilian, Spanish model originally trained by simonestradasch.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/nercomlower_bert_base_spanish_wwm_cased_pipeline_es_5.5.1_3.0_1738842097524.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/nercomlower_bert_base_spanish_wwm_cased_pipeline_es_5.5.1_3.0_1738842097524.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("nercomlower_bert_base_spanish_wwm_cased_pipeline", lang = "es")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("nercomlower_bert_base_spanish_wwm_cased_pipeline", lang = "es")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|nercomlower_bert_base_spanish_wwm_cased_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|es|
|Size:|409.6 MB|

## References

https://huggingface.co/simonestradasch/nercomlower-bert-base-spanish-wwm-cased

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification