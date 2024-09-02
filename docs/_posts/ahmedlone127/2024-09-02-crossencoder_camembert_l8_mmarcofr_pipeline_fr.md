---
layout: model
title: French crossencoder_camembert_l8_mmarcofr_pipeline pipeline CamemBertForSequenceClassification from antoinelouis
author: John Snow Labs
name: crossencoder_camembert_l8_mmarcofr_pipeline
date: 2024-09-02
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

Pretrained CamemBertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`crossencoder_camembert_l8_mmarcofr_pipeline` is a French model originally trained by antoinelouis.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/crossencoder_camembert_l8_mmarcofr_pipeline_fr_5.5.0_3.0_1725298329326.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/crossencoder_camembert_l8_mmarcofr_pipeline_fr_5.5.0_3.0_1725298329326.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("crossencoder_camembert_l8_mmarcofr_pipeline", lang = "fr")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("crossencoder_camembert_l8_mmarcofr_pipeline", lang = "fr")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|crossencoder_camembert_l8_mmarcofr_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|fr|
|Size:|308.5 MB|

## References

https://huggingface.co/antoinelouis/crossencoder-camembert-L8-mmarcoFR

## Included Models

- DocumentAssembler
- TokenizerModel
- CamemBertForSequenceClassification