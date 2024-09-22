---
layout: model
title: English sead_l_6_h_256_a_8_qqp_pipeline pipeline BertForSequenceClassification from C5i
author: John Snow Labs
name: sead_l_6_h_256_a_8_qqp_pipeline
date: 2024-09-20
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

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sead_l_6_h_256_a_8_qqp_pipeline` is a English model originally trained by C5i.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sead_l_6_h_256_a_8_qqp_pipeline_en_5.5.0_3.0_1726803482245.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sead_l_6_h_256_a_8_qqp_pipeline_en_5.5.0_3.0_1726803482245.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("sead_l_6_h_256_a_8_qqp_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("sead_l_6_h_256_a_8_qqp_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sead_l_6_h_256_a_8_qqp_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|47.4 MB|

## References

https://huggingface.co/C5i/SEAD-L-6_H-256_A-8-qqp

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification