---
layout: model
title: Castilian, Spanish roberta_bne_fine_tuned_text_classification_slovene_dss_pipeline pipeline RoBertaForSequenceClassification from Sleoruiz
author: John Snow Labs
name: roberta_bne_fine_tuned_text_classification_slovene_dss_pipeline
date: 2024-09-09
tags: [es, open_source, pipeline, onnx]
task: Text Classification
language: es
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`roberta_bne_fine_tuned_text_classification_slovene_dss_pipeline` is a Castilian, Spanish model originally trained by Sleoruiz.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_bne_fine_tuned_text_classification_slovene_dss_pipeline_es_5.5.0_3.0_1725911591810.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_bne_fine_tuned_text_classification_slovene_dss_pipeline_es_5.5.0_3.0_1725911591810.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("roberta_bne_fine_tuned_text_classification_slovene_dss_pipeline", lang = "es")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("roberta_bne_fine_tuned_text_classification_slovene_dss_pipeline", lang = "es")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_bne_fine_tuned_text_classification_slovene_dss_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|es|
|Size:|453.8 MB|

## References

https://huggingface.co/Sleoruiz/roberta-bne-fine-tuned-text-classification-SL-dss

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification