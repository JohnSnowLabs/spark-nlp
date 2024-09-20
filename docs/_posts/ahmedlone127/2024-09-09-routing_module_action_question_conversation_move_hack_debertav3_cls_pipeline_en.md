---
layout: model
title: English routing_module_action_question_conversation_move_hack_debertav3_cls_pipeline pipeline DeBertaForSequenceClassification from Raffix
author: John Snow Labs
name: routing_module_action_question_conversation_move_hack_debertav3_cls_pipeline
date: 2024-09-09
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

Pretrained DeBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`routing_module_action_question_conversation_move_hack_debertav3_cls_pipeline` is a English model originally trained by Raffix.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/routing_module_action_question_conversation_move_hack_debertav3_cls_pipeline_en_5.5.0_3.0_1725858830207.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/routing_module_action_question_conversation_move_hack_debertav3_cls_pipeline_en_5.5.0_3.0_1725858830207.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("routing_module_action_question_conversation_move_hack_debertav3_cls_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("routing_module_action_question_conversation_move_hack_debertav3_cls_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|routing_module_action_question_conversation_move_hack_debertav3_cls_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|574.3 MB|

## References

https://huggingface.co/Raffix/routing_module_action_question_conversation_move_hack_debertav3_cls

## Included Models

- DocumentAssembler
- TokenizerModel
- DeBertaForSequenceClassification