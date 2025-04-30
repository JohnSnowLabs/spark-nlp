---
layout: model
title: English gmra_model_distilbert_base_uncased_distilled_squad_17082023t150913_pipeline pipeline DistilBertForSequenceClassification from samaksh-khatri-crest-data
author: John Snow Labs
name: gmra_model_distilbert_base_uncased_distilled_squad_17082023t150913_pipeline
date: 2025-02-03
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

Pretrained DistilBertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`gmra_model_distilbert_base_uncased_distilled_squad_17082023t150913_pipeline` is a English model originally trained by samaksh-khatri-crest-data.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/gmra_model_distilbert_base_uncased_distilled_squad_17082023t150913_pipeline_en_5.5.1_3.0_1738546782104.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/gmra_model_distilbert_base_uncased_distilled_squad_17082023t150913_pipeline_en_5.5.1_3.0_1738546782104.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("gmra_model_distilbert_base_uncased_distilled_squad_17082023t150913_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("gmra_model_distilbert_base_uncased_distilled_squad_17082023t150913_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|gmra_model_distilbert_base_uncased_distilled_squad_17082023t150913_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|249.4 MB|

## References

https://huggingface.co/samaksh-khatri-crest-data/gmra_model_distilbert-base-uncased-distilled-squad_17082023T150913

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertForSequenceClassification