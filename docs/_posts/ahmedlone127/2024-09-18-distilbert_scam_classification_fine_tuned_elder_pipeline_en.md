---
layout: model
title: English distilbert_scam_classification_fine_tuned_elder_pipeline pipeline DistilBertForSequenceClassification from autobio
author: John Snow Labs
name: distilbert_scam_classification_fine_tuned_elder_pipeline
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

Pretrained DistilBertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`distilbert_scam_classification_fine_tuned_elder_pipeline` is a English model originally trained by autobio.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_scam_classification_fine_tuned_elder_pipeline_en_5.5.0_3.0_1726625629004.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distilbert_scam_classification_fine_tuned_elder_pipeline_en_5.5.0_3.0_1726625629004.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("distilbert_scam_classification_fine_tuned_elder_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("distilbert_scam_classification_fine_tuned_elder_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_scam_classification_fine_tuned_elder_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|249.5 MB|

## References

https://huggingface.co/autobio/distilbert-scam-classification-fine-tuned-elder

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertForSequenceClassification