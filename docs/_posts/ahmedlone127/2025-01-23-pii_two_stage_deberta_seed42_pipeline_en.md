---
layout: model
title: English pii_two_stage_deberta_seed42_pipeline pipeline DeBertaForTokenClassification from bogoconic1
author: John Snow Labs
name: pii_two_stage_deberta_seed42_pipeline
date: 2025-01-23
tags: [en, open_source, pipeline, onnx]
task: Named Entity Recognition
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

Pretrained DeBertaForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`pii_two_stage_deberta_seed42_pipeline` is a English model originally trained by bogoconic1.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pii_two_stage_deberta_seed42_pipeline_en_5.5.1_3.0_1737642872374.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/pii_two_stage_deberta_seed42_pipeline_en_5.5.1_3.0_1737642872374.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("pii_two_stage_deberta_seed42_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("pii_two_stage_deberta_seed42_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pii_two_stage_deberta_seed42_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.6 GB|

## References

https://huggingface.co/bogoconic1/pii-two-stage-deberta-seed42

## Included Models

- DocumentAssembler
- TokenizerModel
- DeBertaForTokenClassification