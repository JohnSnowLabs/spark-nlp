---
layout: model
title: Portuguese bertimbau_finetuned_glassdoor_reviews_pipeline pipeline BertForSequenceClassification from stevillis
author: John Snow Labs
name: bertimbau_finetuned_glassdoor_reviews_pipeline
date: 2025-01-28
tags: [pt, open_source, pipeline, onnx]
task: Text Classification
language: pt
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bertimbau_finetuned_glassdoor_reviews_pipeline` is a Portuguese model originally trained by stevillis.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bertimbau_finetuned_glassdoor_reviews_pipeline_pt_5.5.1_3.0_1738101079575.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bertimbau_finetuned_glassdoor_reviews_pipeline_pt_5.5.1_3.0_1738101079575.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bertimbau_finetuned_glassdoor_reviews_pipeline", lang = "pt")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bertimbau_finetuned_glassdoor_reviews_pipeline", lang = "pt")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bertimbau_finetuned_glassdoor_reviews_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|pt|
|Size:|408.2 MB|

## References

https://huggingface.co/stevillis/bertimbau-finetuned-glassdoor-reviews

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification