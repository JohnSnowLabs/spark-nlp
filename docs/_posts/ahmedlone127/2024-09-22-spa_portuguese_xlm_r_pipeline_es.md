---
layout: model
title: Castilian, Spanish spa_portuguese_xlm_r_pipeline pipeline XlmRoBertaForTokenClassification from mbruton
author: John Snow Labs
name: spa_portuguese_xlm_r_pipeline
date: 2024-09-22
tags: [es, open_source, pipeline, onnx]
task: Named Entity Recognition
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

Pretrained XlmRoBertaForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`spa_portuguese_xlm_r_pipeline` is a Castilian, Spanish model originally trained by mbruton.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/spa_portuguese_xlm_r_pipeline_es_5.5.0_3.0_1726970179501.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/spa_portuguese_xlm_r_pipeline_es_5.5.0_3.0_1726970179501.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("spa_portuguese_xlm_r_pipeline", lang = "es")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("spa_portuguese_xlm_r_pipeline", lang = "es")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|spa_portuguese_xlm_r_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|es|
|Size:|865.1 MB|

## References

https://huggingface.co/mbruton/spa_pt_XLM-R

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForTokenClassification