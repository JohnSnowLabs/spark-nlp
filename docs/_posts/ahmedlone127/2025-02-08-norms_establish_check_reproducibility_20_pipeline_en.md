---
layout: model
title: English norms_establish_check_reproducibility_20_pipeline pipeline RoBertaForSequenceClassification from rose-e-wang
author: John Snow Labs
name: norms_establish_check_reproducibility_20_pipeline
date: 2025-02-08
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

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`norms_establish_check_reproducibility_20_pipeline` is a English model originally trained by rose-e-wang.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/norms_establish_check_reproducibility_20_pipeline_en_5.5.1_3.0_1739006090779.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/norms_establish_check_reproducibility_20_pipeline_en_5.5.1_3.0_1739006090779.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("norms_establish_check_reproducibility_20_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("norms_establish_check_reproducibility_20_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|norms_establish_check_reproducibility_20_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.3 GB|

## References

https://huggingface.co/rose-e-wang/norms_establish_check_reproducibility_20

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification