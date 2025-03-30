---
layout: model
title: English social_media_sanskrit_saskta_finetuned_1_pipeline pipeline DistilBertForSequenceClassification from Kwaku
author: John Snow Labs
name: social_media_sanskrit_saskta_finetuned_1_pipeline
date: 2025-03-30
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

Pretrained DistilBertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`social_media_sanskrit_saskta_finetuned_1_pipeline` is a English model originally trained by Kwaku.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/social_media_sanskrit_saskta_finetuned_1_pipeline_en_5.5.1_3.0_1743319338789.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/social_media_sanskrit_saskta_finetuned_1_pipeline_en_5.5.1_3.0_1743319338789.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("social_media_sanskrit_saskta_finetuned_1_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("social_media_sanskrit_saskta_finetuned_1_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|social_media_sanskrit_saskta_finetuned_1_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|249.5 MB|

## References

https://huggingface.co/Kwaku/social_media_sa_finetuned_1

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertForSequenceClassification