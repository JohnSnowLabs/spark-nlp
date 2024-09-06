---
layout: model
title: English autotrain_okr_iptal_v5_48523117787_pipeline pipeline XlmRoBertaForSequenceClassification from ekincanozcelik
author: John Snow Labs
name: autotrain_okr_iptal_v5_48523117787_pipeline
date: 2024-09-06
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

Pretrained XlmRoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`autotrain_okr_iptal_v5_48523117787_pipeline` is a English model originally trained by ekincanozcelik.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/autotrain_okr_iptal_v5_48523117787_pipeline_en_5.5.0_3.0_1725619838965.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/autotrain_okr_iptal_v5_48523117787_pipeline_en_5.5.0_3.0_1725619838965.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("autotrain_okr_iptal_v5_48523117787_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("autotrain_okr_iptal_v5_48523117787_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|autotrain_okr_iptal_v5_48523117787_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|778.4 MB|

## References

https://huggingface.co/ekincanozcelik/autotrain-okr_iptal_v5-48523117787

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForSequenceClassification