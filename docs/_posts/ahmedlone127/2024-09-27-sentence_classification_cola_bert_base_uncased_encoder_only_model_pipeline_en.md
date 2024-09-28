---
layout: model
title: English sentence_classification_cola_bert_base_uncased_encoder_only_model_pipeline pipeline BertForSequenceClassification from Dwaraka
author: John Snow Labs
name: sentence_classification_cola_bert_base_uncased_encoder_only_model_pipeline
date: 2024-09-27
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

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sentence_classification_cola_bert_base_uncased_encoder_only_model_pipeline` is a English model originally trained by Dwaraka.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sentence_classification_cola_bert_base_uncased_encoder_only_model_pipeline_en_5.5.0_3.0_1727413139011.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sentence_classification_cola_bert_base_uncased_encoder_only_model_pipeline_en_5.5.0_3.0_1727413139011.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("sentence_classification_cola_bert_base_uncased_encoder_only_model_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("sentence_classification_cola_bert_base_uncased_encoder_only_model_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sentence_classification_cola_bert_base_uncased_encoder_only_model_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|409.4 MB|

## References

https://huggingface.co/Dwaraka/Sentence_Classification_CoLA_BERT_base_uncased_Encoder_Only_Model

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification