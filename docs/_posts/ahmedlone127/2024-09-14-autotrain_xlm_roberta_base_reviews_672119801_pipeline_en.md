---
layout: model
title: English autotrain_xlm_roberta_base_reviews_672119801_pipeline pipeline XlmRoBertaForSequenceClassification from YXHugging
author: John Snow Labs
name: autotrain_xlm_roberta_base_reviews_672119801_pipeline
date: 2024-09-14
tags: [en, open_source, pipeline, onnx]
task: Text Classification
language: en
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`autotrain_xlm_roberta_base_reviews_672119801_pipeline` is a English model originally trained by YXHugging.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/autotrain_xlm_roberta_base_reviews_672119801_pipeline_en_5.5.0_3.0_1726353698717.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/autotrain_xlm_roberta_base_reviews_672119801_pipeline_en_5.5.0_3.0_1726353698717.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("autotrain_xlm_roberta_base_reviews_672119801_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("autotrain_xlm_roberta_base_reviews_672119801_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|autotrain_xlm_roberta_base_reviews_672119801_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|856.7 MB|

## References

https://huggingface.co/YXHugging/autotrain-xlm-roberta-base-reviews-672119801

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForSequenceClassification