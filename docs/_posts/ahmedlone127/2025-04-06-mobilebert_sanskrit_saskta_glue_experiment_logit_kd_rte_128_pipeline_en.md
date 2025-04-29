---
layout: model
title: English mobilebert_sanskrit_saskta_glue_experiment_logit_kd_rte_128_pipeline pipeline BertForSequenceClassification from gokuls
author: John Snow Labs
name: mobilebert_sanskrit_saskta_glue_experiment_logit_kd_rte_128_pipeline
date: 2025-04-06
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

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mobilebert_sanskrit_saskta_glue_experiment_logit_kd_rte_128_pipeline` is a English model originally trained by gokuls.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mobilebert_sanskrit_saskta_glue_experiment_logit_kd_rte_128_pipeline_en_5.5.1_3.0_1743964087129.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mobilebert_sanskrit_saskta_glue_experiment_logit_kd_rte_128_pipeline_en_5.5.1_3.0_1743964087129.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("mobilebert_sanskrit_saskta_glue_experiment_logit_kd_rte_128_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("mobilebert_sanskrit_saskta_glue_experiment_logit_kd_rte_128_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mobilebert_sanskrit_saskta_glue_experiment_logit_kd_rte_128_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|73.3 MB|

## References

https://huggingface.co/gokuls/mobilebert_sa_GLUE_Experiment_logit_kd_rte_128

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification