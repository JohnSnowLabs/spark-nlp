---
layout: model
title: English bert_base_cased_5kvul_10aug_3nsfw_10w_exp_10ep_pipeline pipeline BertForSequenceClassification from AbhishekkV19
author: John Snow Labs
name: bert_base_cased_5kvul_10aug_3nsfw_10w_exp_10ep_pipeline
date: 2024-09-26
tags: [en, open_source, pipeline, onnx]
task: Text Classification
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

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bert_base_cased_5kvul_10aug_3nsfw_10w_exp_10ep_pipeline` is a English model originally trained by AbhishekkV19.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_base_cased_5kvul_10aug_3nsfw_10w_exp_10ep_pipeline_en_5.5.0_3.0_1727387018791.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_base_cased_5kvul_10aug_3nsfw_10w_exp_10ep_pipeline_en_5.5.0_3.0_1727387018791.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bert_base_cased_5kvul_10aug_3nsfw_10w_exp_10ep_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bert_base_cased_5kvul_10aug_3nsfw_10w_exp_10ep_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_base_cased_5kvul_10aug_3nsfw_10w_exp_10ep_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|405.9 MB|

## References

https://huggingface.co/AbhishekkV19/bert-base-cased-5kvul-10aug-3nsfw-10w-exp-10ep

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification