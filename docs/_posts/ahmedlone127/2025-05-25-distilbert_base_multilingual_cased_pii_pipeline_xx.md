---
layout: model
title: Multilingual distilbert_base_multilingual_cased_pii_pipeline pipeline DistilBertForTokenClassification from yonigo
author: John Snow Labs
name: distilbert_base_multilingual_cased_pii_pipeline
date: 2025-05-25
tags: [xx, open_source, pipeline, onnx]
task: Named Entity Recognition
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

Pretrained DistilBertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`distilbert_base_multilingual_cased_pii_pipeline` is a Multilingual model originally trained by yonigo.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_base_multilingual_cased_pii_pipeline_xx_5.5.1_3.0_1748182936635.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distilbert_base_multilingual_cased_pii_pipeline_xx_5.5.1_3.0_1748182936635.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("distilbert_base_multilingual_cased_pii_pipeline", lang = "xx")
annotations =  pipeline.transform(df)
```
```scala
val pipeline = new PretrainedPipeline("distilbert_base_multilingual_cased_pii_pipeline", lang = "xx")
val annotations = pipeline.transform(df)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_base_multilingual_cased_pii_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|xx|
|Size:|505.6 MB|

## References

References

https://huggingface.co/yonigo/distilbert-base-multilingual-cased-pii

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertForTokenClassification