---
layout: model
title: English robertuito_4clases_pfinal_pipeline pipeline RoBertaForSequenceClassification from fredymad
author: John Snow Labs
name: robertuito_4clases_pfinal_pipeline
date: 2024-09-10
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

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`robertuito_4clases_pfinal_pipeline` is a English model originally trained by fredymad.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/robertuito_4clases_pfinal_pipeline_en_5.5.0_3.0_1725972183503.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/robertuito_4clases_pfinal_pipeline_en_5.5.0_3.0_1725972183503.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("robertuito_4clases_pfinal_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("robertuito_4clases_pfinal_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|robertuito_4clases_pfinal_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|408.3 MB|

## References

https://huggingface.co/fredymad/robertuito_4CLASES_Pfinal

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification