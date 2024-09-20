---
layout: model
title: English all_mpnet_base_v2__tweet_eval_emotion__classifier_pipeline pipeline MPNetForSequenceClassification from florentgbelidji
author: John Snow Labs
name: all_mpnet_base_v2__tweet_eval_emotion__classifier_pipeline
date: 2024-09-06
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

Pretrained MPNetForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`all_mpnet_base_v2__tweet_eval_emotion__classifier_pipeline` is a English model originally trained by florentgbelidji.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/all_mpnet_base_v2__tweet_eval_emotion__classifier_pipeline_en_5.5.0_3.0_1725655765622.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/all_mpnet_base_v2__tweet_eval_emotion__classifier_pipeline_en_5.5.0_3.0_1725655765622.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("all_mpnet_base_v2__tweet_eval_emotion__classifier_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("all_mpnet_base_v2__tweet_eval_emotion__classifier_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|all_mpnet_base_v2__tweet_eval_emotion__classifier_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|409.4 MB|

## References

https://huggingface.co/florentgbelidji/all-mpnet-base-v2__tweet_eval_emotion__classifier

## Included Models

- DocumentAssembler
- TokenizerModel
- MPNetForSequenceClassification