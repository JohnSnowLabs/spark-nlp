---
layout: model
title: Afar bert_fine_tunned_hate_speech_pipeline pipeline DistilBertForSequenceClassification from zenitsu0509
author: John Snow Labs
name: bert_fine_tunned_hate_speech_pipeline
date: 2024-10-09
tags: [aa, open_source, pipeline, onnx]
task: Text Classification
language: aa
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DistilBertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bert_fine_tunned_hate_speech_pipeline` is a Afar model originally trained by zenitsu0509.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_fine_tunned_hate_speech_pipeline_aa_5.5.1_3.0_1728457409632.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_fine_tunned_hate_speech_pipeline_aa_5.5.1_3.0_1728457409632.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bert_fine_tunned_hate_speech_pipeline", lang = "aa")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bert_fine_tunned_hate_speech_pipeline", lang = "aa")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_fine_tunned_hate_speech_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|aa|
|Size:|408.1 MB|

## References

https://huggingface.co/zenitsu0509/bert_fine_tunned_hate_speech

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertForSequenceClassification