---
layout: model
title: Korean ableism_miner_revised_pipeline pipeline BertForSequenceClassification from NeinYeop
author: John Snow Labs
name: ableism_miner_revised_pipeline
date: 2024-10-09
tags: [ko, open_source, pipeline, onnx]
task: Text Classification
language: ko
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`ableism_miner_revised_pipeline` is a Korean model originally trained by NeinYeop.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ableism_miner_revised_pipeline_ko_5.5.1_3.0_1728451534566.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/ableism_miner_revised_pipeline_ko_5.5.1_3.0_1728451534566.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("ableism_miner_revised_pipeline", lang = "ko")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("ableism_miner_revised_pipeline", lang = "ko")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ableism_miner_revised_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|ko|
|Size:|408.5 MB|

## References

https://huggingface.co/NeinYeop/ableism-miner_revised

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification