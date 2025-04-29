---
layout: model
title: English bert_base_uncased_finetune_squad_ep_2_0_lr_0_0005_wd_0_1_dp_0_33_pipeline pipeline BertForQuestionAnswering from danielkty22
author: John Snow Labs
name: bert_base_uncased_finetune_squad_ep_2_0_lr_0_0005_wd_0_1_dp_0_33_pipeline
date: 2025-01-30
tags: [en, open_source, pipeline, onnx]
task: Question Answering
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

Pretrained BertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bert_base_uncased_finetune_squad_ep_2_0_lr_0_0005_wd_0_1_dp_0_33_pipeline` is a English model originally trained by danielkty22.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_base_uncased_finetune_squad_ep_2_0_lr_0_0005_wd_0_1_dp_0_33_pipeline_en_5.5.1_3.0_1738223712481.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_base_uncased_finetune_squad_ep_2_0_lr_0_0005_wd_0_1_dp_0_33_pipeline_en_5.5.1_3.0_1738223712481.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bert_base_uncased_finetune_squad_ep_2_0_lr_0_0005_wd_0_1_dp_0_33_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bert_base_uncased_finetune_squad_ep_2_0_lr_0_0005_wd_0_1_dp_0_33_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_base_uncased_finetune_squad_ep_2_0_lr_0_0005_wd_0_1_dp_0_33_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|407.3 MB|

## References

https://huggingface.co/danielkty22/bert-base-uncased-finetune-squad-ep-2.0-lr-0.0005-wd-0.1-dp-0.33

## Included Models

- MultiDocumentAssembler
- BertForQuestionAnswering