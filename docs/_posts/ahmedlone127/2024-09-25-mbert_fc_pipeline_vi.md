---
layout: model
title: Vietnamese mbert_fc_pipeline pipeline BertForSequenceClassification from SonFox2920
author: John Snow Labs
name: mbert_fc_pipeline
date: 2024-09-25
tags: [vi, open_source, pipeline, onnx]
task: Text Classification
language: vi
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mbert_fc_pipeline` is a Vietnamese model originally trained by SonFox2920.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mbert_fc_pipeline_vi_5.5.0_3.0_1727290023120.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mbert_fc_pipeline_vi_5.5.0_3.0_1727290023120.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("mbert_fc_pipeline", lang = "vi")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("mbert_fc_pipeline", lang = "vi")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mbert_fc_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|vi|
|Size:|667.3 MB|

## References

https://huggingface.co/SonFox2920/MBert_FC

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification