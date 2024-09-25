---
layout: model
title: English aes_bert_base_sp90_lr1e_05_wr1e_01_wd1e_02_ep15_elsa_pipeline pipeline BertForSequenceClassification from ys7yoo
author: John Snow Labs
name: aes_bert_base_sp90_lr1e_05_wr1e_01_wd1e_02_ep15_elsa_pipeline
date: 2024-09-25
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

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`aes_bert_base_sp90_lr1e_05_wr1e_01_wd1e_02_ep15_elsa_pipeline` is a English model originally trained by ys7yoo.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/aes_bert_base_sp90_lr1e_05_wr1e_01_wd1e_02_ep15_elsa_pipeline_en_5.5.0_3.0_1727287956799.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/aes_bert_base_sp90_lr1e_05_wr1e_01_wd1e_02_ep15_elsa_pipeline_en_5.5.0_3.0_1727287956799.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("aes_bert_base_sp90_lr1e_05_wr1e_01_wd1e_02_ep15_elsa_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("aes_bert_base_sp90_lr1e_05_wr1e_01_wd1e_02_ep15_elsa_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|aes_bert_base_sp90_lr1e_05_wr1e_01_wd1e_02_ep15_elsa_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|414.6 MB|

## References

https://huggingface.co/ys7yoo/aes_bert-base_sp90_lr1e-05_wr1e-01_wd1e-02_ep15_elsa

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification