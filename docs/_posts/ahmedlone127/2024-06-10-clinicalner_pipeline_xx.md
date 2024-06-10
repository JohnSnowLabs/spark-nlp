---
layout: model
title: Multilingual clinicalner_pipeline pipeline XlmRoBertaForTokenClassification from Posos
author: John Snow Labs
name: clinicalner_pipeline
date: 2024-06-10
tags: [xx, open_source, pipeline, onnx]
task: Named Entity Recognition
language: xx
edition: Spark NLP 5.4.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`clinicalner_pipeline` is a Multilingual model originally trained by Posos.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/clinicalner_pipeline_xx_5.4.0_3.0_1718041543627.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/clinicalner_pipeline_xx_5.4.0_3.0_1718041543627.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("clinicalner_pipeline", lang = "xx")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("clinicalner_pipeline", lang = "xx")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|clinicalner_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|xx|
|Size:|784.8 MB|

## References

https://huggingface.co/Posos/ClinicalNER

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForTokenClassification