---
layout: model
title: English merged_bert_sst2_model_lora_pipeline pipeline BertForSequenceClassification from asm3515
author: John Snow Labs
name: merged_bert_sst2_model_lora_pipeline
date: 2025-01-25
tags: [en, open_source, pipeline, onnx]
task: Text Classification
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

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`merged_bert_sst2_model_lora_pipeline` is a English model originally trained by asm3515.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/merged_bert_sst2_model_lora_pipeline_en_5.5.1_3.0_1737800322917.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/merged_bert_sst2_model_lora_pipeline_en_5.5.1_3.0_1737800322917.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("merged_bert_sst2_model_lora_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("merged_bert_sst2_model_lora_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|merged_bert_sst2_model_lora_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|409.4 MB|

## References

https://huggingface.co/asm3515/merged-bert-sst2-model-lora

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification