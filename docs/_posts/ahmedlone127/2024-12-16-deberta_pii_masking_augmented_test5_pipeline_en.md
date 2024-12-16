---
layout: model
title: English deberta_pii_masking_augmented_test5_pipeline pipeline DeBertaForTokenClassification from gaodrew
author: John Snow Labs
name: deberta_pii_masking_augmented_test5_pipeline
date: 2024-12-16
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

Pretrained DeBertaForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`deberta_pii_masking_augmented_test5_pipeline` is a English model originally trained by gaodrew.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/deberta_pii_masking_augmented_test5_pipeline_en_5.5.1_3.0_1734347202028.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/deberta_pii_masking_augmented_test5_pipeline_en_5.5.1_3.0_1734347202028.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("deberta_pii_masking_augmented_test5_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("deberta_pii_masking_augmented_test5_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|deberta_pii_masking_augmented_test5_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.5 GB|

## References

https://huggingface.co/gaodrew/deberta-pii-masking-augmented-test5

## Included Models

- DocumentAssembler
- TokenizerModel
- DeBertaForTokenClassification