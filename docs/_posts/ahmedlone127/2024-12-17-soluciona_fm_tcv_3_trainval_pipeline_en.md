---
layout: model
title: English soluciona_fm_tcv_3_trainval_pipeline pipeline XlmRoBertaForSequenceClassification from adriansanz
author: John Snow Labs
name: soluciona_fm_tcv_3_trainval_pipeline
date: 2024-12-17
tags: [en, open_source, pipeline, onnx]
task: Text Classification
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`soluciona_fm_tcv_3_trainval_pipeline` is a English model originally trained by adriansanz.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/soluciona_fm_tcv_3_trainval_pipeline_en_5.5.1_3.0_1734418805367.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/soluciona_fm_tcv_3_trainval_pipeline_en_5.5.1_3.0_1734418805367.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("soluciona_fm_tcv_3_trainval_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("soluciona_fm_tcv_3_trainval_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|soluciona_fm_tcv_3_trainval_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|778.5 MB|

## References

https://huggingface.co/adriansanz/soluciona_fm_tcv_3_trainval

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForSequenceClassification