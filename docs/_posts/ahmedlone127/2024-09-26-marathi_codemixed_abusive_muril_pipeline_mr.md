---
layout: model
title: Marathi marathi_codemixed_abusive_muril_pipeline pipeline BertForSequenceClassification from Hate-speech-CNERG
author: John Snow Labs
name: marathi_codemixed_abusive_muril_pipeline
date: 2024-09-26
tags: [mr, open_source, pipeline, onnx]
task: Text Classification
language: mr
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`marathi_codemixed_abusive_muril_pipeline` is a Marathi model originally trained by Hate-speech-CNERG.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/marathi_codemixed_abusive_muril_pipeline_mr_5.5.0_3.0_1727345137386.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/marathi_codemixed_abusive_muril_pipeline_mr_5.5.0_3.0_1727345137386.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("marathi_codemixed_abusive_muril_pipeline", lang = "mr")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("marathi_codemixed_abusive_muril_pipeline", lang = "mr")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|marathi_codemixed_abusive_muril_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|mr|
|Size:|892.7 MB|

## References

https://huggingface.co/Hate-speech-CNERG/marathi-codemixed-abusive-MuRIL

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification