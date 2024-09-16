---
layout: model
title: English imdbreviews_classification_roberta_v01_clf_finetuning_dfelorza_pipeline pipeline RoBertaForSequenceClassification from dfelorza
author: John Snow Labs
name: imdbreviews_classification_roberta_v01_clf_finetuning_dfelorza_pipeline
date: 2024-09-13
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

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`imdbreviews_classification_roberta_v01_clf_finetuning_dfelorza_pipeline` is a English model originally trained by dfelorza.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/imdbreviews_classification_roberta_v01_clf_finetuning_dfelorza_pipeline_en_5.5.0_3.0_1726227329308.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/imdbreviews_classification_roberta_v01_clf_finetuning_dfelorza_pipeline_en_5.5.0_3.0_1726227329308.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("imdbreviews_classification_roberta_v01_clf_finetuning_dfelorza_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("imdbreviews_classification_roberta_v01_clf_finetuning_dfelorza_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|imdbreviews_classification_roberta_v01_clf_finetuning_dfelorza_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|300.4 MB|

## References

https://huggingface.co/dfelorza/imdbreviews_classification_roberta_v01_clf_finetuning

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification