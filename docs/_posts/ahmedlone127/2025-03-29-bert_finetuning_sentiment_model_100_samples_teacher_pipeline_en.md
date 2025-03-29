---
layout: model
title: English bert_finetuning_sentiment_model_100_samples_teacher_pipeline pipeline DistilBertForSequenceClassification from skrh
author: John Snow Labs
name: bert_finetuning_sentiment_model_100_samples_teacher_pipeline
date: 2025-03-29
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

Pretrained DistilBertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bert_finetuning_sentiment_model_100_samples_teacher_pipeline` is a English model originally trained by skrh.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_finetuning_sentiment_model_100_samples_teacher_pipeline_en_5.5.1_3.0_1743265613965.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_finetuning_sentiment_model_100_samples_teacher_pipeline_en_5.5.1_3.0_1743265613965.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bert_finetuning_sentiment_model_100_samples_teacher_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bert_finetuning_sentiment_model_100_samples_teacher_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_finetuning_sentiment_model_100_samples_teacher_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|249.5 MB|

## References

https://huggingface.co/skrh/bert_finetuning-sentiment-model-100-samples-teacher

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertForSequenceClassification