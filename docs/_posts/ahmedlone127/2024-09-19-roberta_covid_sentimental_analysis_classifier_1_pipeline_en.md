---
layout: model
title: English roberta_covid_sentimental_analysis_classifier_1_pipeline pipeline RoBertaForSequenceClassification from gyesibiney
author: John Snow Labs
name: roberta_covid_sentimental_analysis_classifier_1_pipeline
date: 2024-09-19
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

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`roberta_covid_sentimental_analysis_classifier_1_pipeline` is a English model originally trained by gyesibiney.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_covid_sentimental_analysis_classifier_1_pipeline_en_5.5.0_3.0_1726750323622.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_covid_sentimental_analysis_classifier_1_pipeline_en_5.5.0_3.0_1726750323622.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("roberta_covid_sentimental_analysis_classifier_1_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("roberta_covid_sentimental_analysis_classifier_1_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_covid_sentimental_analysis_classifier_1_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|468.2 MB|

## References

https://huggingface.co/gyesibiney/roberta-covid-sentimental-analysis-classifier-1

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification