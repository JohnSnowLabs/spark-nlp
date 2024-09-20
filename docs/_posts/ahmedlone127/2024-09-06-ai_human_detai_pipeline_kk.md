---
layout: model
title: Kazakh ai_human_detai_pipeline pipeline DistilBertForSequenceClassification from Ayanm
author: John Snow Labs
name: ai_human_detai_pipeline
date: 2024-09-06
tags: [kk, open_source, pipeline, onnx]
task: Text Classification
language: kk
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DistilBertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`ai_human_detai_pipeline` is a Kazakh model originally trained by Ayanm.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ai_human_detai_pipeline_kk_5.5.0_3.0_1725608332378.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/ai_human_detai_pipeline_kk_5.5.0_3.0_1725608332378.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("ai_human_detai_pipeline", lang = "kk")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("ai_human_detai_pipeline", lang = "kk")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ai_human_detai_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|kk|
|Size:|249.5 MB|

## References

https://huggingface.co/Ayanm/ai-human-detai

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertForSequenceClassification