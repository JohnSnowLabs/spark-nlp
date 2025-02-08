---
layout: model
title: English claim_verification_xlm_r_pipeline pipeline XlmRoBertaForSequenceClassification from Blue7Bird
author: John Snow Labs
name: claim_verification_xlm_r_pipeline
date: 2025-02-05
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

Pretrained XlmRoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`claim_verification_xlm_r_pipeline` is a English model originally trained by Blue7Bird.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/claim_verification_xlm_r_pipeline_en_5.5.1_3.0_1738793614889.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/claim_verification_xlm_r_pipeline_en_5.5.1_3.0_1738793614889.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("claim_verification_xlm_r_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("claim_verification_xlm_r_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|claim_verification_xlm_r_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|655.2 MB|

## References

https://huggingface.co/Blue7Bird/Claim_verification_XLM_R

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForSequenceClassification