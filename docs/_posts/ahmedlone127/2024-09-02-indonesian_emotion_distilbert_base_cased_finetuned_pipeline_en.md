---
layout: model
title: English indonesian_emotion_distilbert_base_cased_finetuned_pipeline pipeline DistilBertForSequenceClassification from AptaArkana
author: John Snow Labs
name: indonesian_emotion_distilbert_base_cased_finetuned_pipeline
date: 2024-09-02
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

Pretrained DistilBertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`indonesian_emotion_distilbert_base_cased_finetuned_pipeline` is a English model originally trained by AptaArkana.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/indonesian_emotion_distilbert_base_cased_finetuned_pipeline_en_5.5.0_3.0_1725306162353.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/indonesian_emotion_distilbert_base_cased_finetuned_pipeline_en_5.5.0_3.0_1725306162353.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("indonesian_emotion_distilbert_base_cased_finetuned_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("indonesian_emotion_distilbert_base_cased_finetuned_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|indonesian_emotion_distilbert_base_cased_finetuned_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|507.7 MB|

## References

https://huggingface.co/AptaArkana/indonesian-emotion-distilbert-base-cased-finetuned

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertForSequenceClassification