---
layout: model
title: English t5_e2e_10epochs_lr1e4_alpha0_1plusalpha0_9_e30_pipeline pipeline T5Transformer from harish
author: John Snow Labs
name: t5_e2e_10epochs_lr1e4_alpha0_1plusalpha0_9_e30_pipeline
date: 2024-08-18
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

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`t5_e2e_10epochs_lr1e4_alpha0_1plusalpha0_9_e30_pipeline` is a English model originally trained by harish.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/t5_e2e_10epochs_lr1e4_alpha0_1plusalpha0_9_e30_pipeline_en_5.4.2_3.0_1724007469749.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/t5_e2e_10epochs_lr1e4_alpha0_1plusalpha0_9_e30_pipeline_en_5.4.2_3.0_1724007469749.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("t5_e2e_10epochs_lr1e4_alpha0_1plusalpha0_9_e30_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("t5_e2e_10epochs_lr1e4_alpha0_1plusalpha0_9_e30_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|t5_e2e_10epochs_lr1e4_alpha0_1plusalpha0_9_e30_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|993.2 MB|

## References

https://huggingface.co/harish/t5-e2e-10epochs-lr1e4-alpha0-1PLUSalpha0-9-e30

## Included Models

- DocumentAssembler
- T5Transformer