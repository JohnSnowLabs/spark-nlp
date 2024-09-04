---
layout: model
title: French monobert_legal_french_pipeline pipeline CamemBertForSequenceClassification from maastrichtlawtech
author: John Snow Labs
name: monobert_legal_french_pipeline
date: 2024-09-04
tags: [fr, open_source, pipeline, onnx]
task: Text Classification
language: fr
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained CamemBertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`monobert_legal_french_pipeline` is a French model originally trained by maastrichtlawtech.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/monobert_legal_french_pipeline_fr_5.5.0_3.0_1725466675822.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/monobert_legal_french_pipeline_fr_5.5.0_3.0_1725466675822.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("monobert_legal_french_pipeline", lang = "fr")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("monobert_legal_french_pipeline", lang = "fr")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|monobert_legal_french_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|fr|
|Size:|415.0 MB|

## References

https://huggingface.co/maastrichtlawtech/monobert-legal-french

## Included Models

- DocumentAssembler
- TokenizerModel
- CamemBertForSequenceClassification