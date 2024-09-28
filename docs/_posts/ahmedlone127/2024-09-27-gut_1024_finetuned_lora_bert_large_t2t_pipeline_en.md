---
layout: model
title: English gut_1024_finetuned_lora_bert_large_t2t_pipeline pipeline BertForSequenceClassification from LiukG
author: John Snow Labs
name: gut_1024_finetuned_lora_bert_large_t2t_pipeline
date: 2024-09-27
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

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`gut_1024_finetuned_lora_bert_large_t2t_pipeline` is a English model originally trained by LiukG.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/gut_1024_finetuned_lora_bert_large_t2t_pipeline_en_5.5.0_3.0_1727416273465.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/gut_1024_finetuned_lora_bert_large_t2t_pipeline_en_5.5.0_3.0_1727416273465.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("gut_1024_finetuned_lora_bert_large_t2t_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("gut_1024_finetuned_lora_bert_large_t2t_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|gut_1024_finetuned_lora_bert_large_t2t_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.0 GB|

## References

https://huggingface.co/LiukG/gut_1024-finetuned-lora-bert-large-t2t

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification