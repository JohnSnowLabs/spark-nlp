---
layout: model
title: Multilingual xlmr_base_refusal_classifier_pipeline pipeline XlmRoBertaForSequenceClassification from chameleon-lizard
author: John Snow Labs
name: xlmr_base_refusal_classifier_pipeline
date: 2025-01-25
tags: [xx, open_source, pipeline, onnx]
task: Text Classification
language: xx
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`xlmr_base_refusal_classifier_pipeline` is a Multilingual model originally trained by chameleon-lizard.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlmr_base_refusal_classifier_pipeline_xx_5.5.1_3.0_1737815328241.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xlmr_base_refusal_classifier_pipeline_xx_5.5.1_3.0_1737815328241.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("xlmr_base_refusal_classifier_pipeline", lang = "xx")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("xlmr_base_refusal_classifier_pipeline", lang = "xx")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlmr_base_refusal_classifier_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|xx|
|Size:|845.0 MB|

## References

https://huggingface.co/chameleon-lizard/xlmr-base-refusal-classifier

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForSequenceClassification