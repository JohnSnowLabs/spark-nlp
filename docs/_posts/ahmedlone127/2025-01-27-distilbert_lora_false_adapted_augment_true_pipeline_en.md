---
layout: model
title: English distilbert_lora_false_adapted_augment_true_pipeline pipeline DistilBertForSequenceClassification from EmiMule
author: John Snow Labs
name: distilbert_lora_false_adapted_augment_true_pipeline
date: 2025-01-27
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

Pretrained DistilBertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`distilbert_lora_false_adapted_augment_true_pipeline` is a English model originally trained by EmiMule.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_lora_false_adapted_augment_true_pipeline_en_5.5.1_3.0_1737939228506.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distilbert_lora_false_adapted_augment_true_pipeline_en_5.5.1_3.0_1737939228506.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("distilbert_lora_false_adapted_augment_true_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("distilbert_lora_false_adapted_augment_true_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_lora_false_adapted_augment_true_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|249.5 MB|

## References

https://huggingface.co/EmiMule/distilbert-LoRA-False-adapted-augment-True

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertForSequenceClassification