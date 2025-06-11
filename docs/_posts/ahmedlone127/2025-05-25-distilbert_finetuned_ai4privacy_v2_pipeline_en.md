---
layout: model
title: English distilbert_finetuned_ai4privacy_v2_pipeline pipeline DistilBertForTokenClassification from Isotonic
author: John Snow Labs
name: distilbert_finetuned_ai4privacy_v2_pipeline
date: 2025-05-25
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

Pretrained DistilBertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`distilbert_finetuned_ai4privacy_v2_pipeline` is a English model originally trained by Isotonic.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_finetuned_ai4privacy_v2_pipeline_en_5.5.1_3.0_1748182935955.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distilbert_finetuned_ai4privacy_v2_pipeline_en_5.5.1_3.0_1748182935955.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("distilbert_finetuned_ai4privacy_v2_pipeline", lang = "en")
annotations =  pipeline.transform(df)
```
```scala
val pipeline = new PretrainedPipeline("distilbert_finetuned_ai4privacy_v2_pipeline", lang = "en")
val annotations = pipeline.transform(df)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_finetuned_ai4privacy_v2_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|247.6 MB|

## References

References

https://huggingface.co/Isotonic/distilbert_finetuned_ai4privacy_v2

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertForTokenClassification