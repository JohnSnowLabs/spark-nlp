---
layout: model
title: Dutch, Flemish echocardiogram_latvian_dilation_reduced_pipeline pipeline RoBertaForSequenceClassification from UMCU
author: John Snow Labs
name: echocardiogram_latvian_dilation_reduced_pipeline
date: 2024-09-22
tags: [nl, open_source, pipeline, onnx]
task: Text Classification
language: nl
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`echocardiogram_latvian_dilation_reduced_pipeline` is a Dutch, Flemish model originally trained by UMCU.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/echocardiogram_latvian_dilation_reduced_pipeline_nl_5.5.0_3.0_1727017617462.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/echocardiogram_latvian_dilation_reduced_pipeline_nl_5.5.0_3.0_1727017617462.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("echocardiogram_latvian_dilation_reduced_pipeline", lang = "nl")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("echocardiogram_latvian_dilation_reduced_pipeline", lang = "nl")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|echocardiogram_latvian_dilation_reduced_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|nl|
|Size:|472.0 MB|

## References

https://huggingface.co/UMCU/Echocardiogram_LV_dilation_reduced

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification