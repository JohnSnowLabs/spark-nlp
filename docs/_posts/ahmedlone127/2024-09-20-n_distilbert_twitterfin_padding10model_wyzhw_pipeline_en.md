---
layout: model
title: English n_distilbert_twitterfin_padding10model_wyzhw_pipeline pipeline DistilBertForSequenceClassification from wyzhw
author: John Snow Labs
name: n_distilbert_twitterfin_padding10model_wyzhw_pipeline
date: 2024-09-20
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

Pretrained DistilBertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`n_distilbert_twitterfin_padding10model_wyzhw_pipeline` is a English model originally trained by wyzhw.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/n_distilbert_twitterfin_padding10model_wyzhw_pipeline_en_5.5.0_3.0_1726860773633.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/n_distilbert_twitterfin_padding10model_wyzhw_pipeline_en_5.5.0_3.0_1726860773633.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("n_distilbert_twitterfin_padding10model_wyzhw_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("n_distilbert_twitterfin_padding10model_wyzhw_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|n_distilbert_twitterfin_padding10model_wyzhw_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|249.5 MB|

## References

https://huggingface.co/wyzhw/N_distilbert_twitterfin_padding10model

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertForSequenceClassification