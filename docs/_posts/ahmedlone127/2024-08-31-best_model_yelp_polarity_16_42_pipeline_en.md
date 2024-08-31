---
layout: model
title: English best_model_yelp_polarity_16_42_pipeline pipeline AlbertForSequenceClassification from simonycl
author: John Snow Labs
name: best_model_yelp_polarity_16_42_pipeline
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

Pretrained AlbertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`best_model_yelp_polarity_16_42_pipeline` is a English model originally trained by simonycl.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/best_model_yelp_polarity_16_42_pipeline_en_5.4.2_3.0_1725125595483.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/best_model_yelp_polarity_16_42_pipeline_en_5.4.2_3.0_1725125595483.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("best_model_yelp_polarity_16_42_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("best_model_yelp_polarity_16_42_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|best_model_yelp_polarity_16_42_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|44.2 MB|

## References

https://huggingface.co/simonycl/best_model-yelp_polarity-16-42

## Included Models

- DocumentAssembler
- TokenizerModel
- AlbertForSequenceClassification