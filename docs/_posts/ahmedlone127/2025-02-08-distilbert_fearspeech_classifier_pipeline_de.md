---
layout: model
title: German distilbert_fearspeech_classifier_pipeline pipeline DistilBertForSequenceClassification from PatrickSchwabl
author: John Snow Labs
name: distilbert_fearspeech_classifier_pipeline
date: 2025-02-08
tags: [de, open_source, pipeline, onnx]
task: Text Classification
language: de
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DistilBertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`distilbert_fearspeech_classifier_pipeline` is a German model originally trained by PatrickSchwabl.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_fearspeech_classifier_pipeline_de_5.5.1_3.0_1739040891836.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distilbert_fearspeech_classifier_pipeline_de_5.5.1_3.0_1739040891836.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("distilbert_fearspeech_classifier_pipeline", lang = "de")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("distilbert_fearspeech_classifier_pipeline", lang = "de")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_fearspeech_classifier_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|de|
|Size:|252.5 MB|

## References

https://huggingface.co/PatrickSchwabl/distilbert_fearspeech_classifier

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertForSequenceClassification