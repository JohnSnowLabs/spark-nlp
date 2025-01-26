---
layout: model
title: Mongolian efficient_fine_tuning_demo_pipeline pipeline BertForTokenClassification from BillyBek
author: John Snow Labs
name: efficient_fine_tuning_demo_pipeline
date: 2025-01-25
tags: [mn, open_source, pipeline, onnx]
task: Named Entity Recognition
language: mn
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`efficient_fine_tuning_demo_pipeline` is a Mongolian model originally trained by BillyBek.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/efficient_fine_tuning_demo_pipeline_mn_5.5.1_3.0_1737843970912.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/efficient_fine_tuning_demo_pipeline_mn_5.5.1_3.0_1737843970912.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("efficient_fine_tuning_demo_pipeline", lang = "mn")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("efficient_fine_tuning_demo_pipeline", lang = "mn")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|efficient_fine_tuning_demo_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|mn|
|Size:|665.1 MB|

## References

https://huggingface.co/BillyBek/efficient-fine-tuning-demo

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification