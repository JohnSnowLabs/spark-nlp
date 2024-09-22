---
layout: model
title: English horai_medium_17k_roberta_large_30_pipeline pipeline RoBertaForSequenceClassification from stealthwriter
author: John Snow Labs
name: horai_medium_17k_roberta_large_30_pipeline
date: 2024-09-18
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

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`horai_medium_17k_roberta_large_30_pipeline` is a English model originally trained by stealthwriter.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/horai_medium_17k_roberta_large_30_pipeline_en_5.5.0_3.0_1726650578959.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/horai_medium_17k_roberta_large_30_pipeline_en_5.5.0_3.0_1726650578959.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("horai_medium_17k_roberta_large_30_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("horai_medium_17k_roberta_large_30_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|horai_medium_17k_roberta_large_30_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.3 GB|

## References

https://huggingface.co/stealthwriter/HorAI-medium-17k-roberta-large-30

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification