---
layout: model
title: English t5_base_few_shot_k_16_finetuned_squad_seed_0_pipeline pipeline T5Transformer from anas-awadalla
author: John Snow Labs
name: t5_base_few_shot_k_16_finetuned_squad_seed_0_pipeline
date: 2024-08-01
tags: [en, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: en
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`t5_base_few_shot_k_16_finetuned_squad_seed_0_pipeline` is a English model originally trained by anas-awadalla.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/t5_base_few_shot_k_16_finetuned_squad_seed_0_pipeline_en_5.4.2_3.0_1722556715997.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/t5_base_few_shot_k_16_finetuned_squad_seed_0_pipeline_en_5.4.2_3.0_1722556715997.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("t5_base_few_shot_k_16_finetuned_squad_seed_0_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("t5_base_few_shot_k_16_finetuned_squad_seed_0_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|t5_base_few_shot_k_16_finetuned_squad_seed_0_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|933.5 MB|

## References

https://huggingface.co/anas-awadalla/t5-base-few-shot-k-16-finetuned-squad-seed-0

## Included Models

- DocumentAssembler
- T5Transformer