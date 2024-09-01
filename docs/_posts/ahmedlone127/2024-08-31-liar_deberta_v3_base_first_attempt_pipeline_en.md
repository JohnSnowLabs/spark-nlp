---
layout: model
title: English liar_deberta_v3_base_first_attempt_pipeline pipeline DeBertaForSequenceClassification from fidhal-ad
author: John Snow Labs
name: liar_deberta_v3_base_first_attempt_pipeline
date: 2024-08-31
tags: [en, open_source, pipeline, onnx]
task: Text Classification
language: en
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DeBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`liar_deberta_v3_base_first_attempt_pipeline` is a English model originally trained by fidhal-ad.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/liar_deberta_v3_base_first_attempt_pipeline_en_5.4.2_3.0_1725124366109.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/liar_deberta_v3_base_first_attempt_pipeline_en_5.4.2_3.0_1725124366109.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("liar_deberta_v3_base_first_attempt_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("liar_deberta_v3_base_first_attempt_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|liar_deberta_v3_base_first_attempt_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|593.8 MB|

## References

https://huggingface.co/fidhal-ad/liar-deberta-v3-base_first_attempt

## Included Models

- DocumentAssembler
- TokenizerModel
- DeBertaForSequenceClassification