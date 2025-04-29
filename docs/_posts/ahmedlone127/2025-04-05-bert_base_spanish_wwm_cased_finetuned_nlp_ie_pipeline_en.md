---
layout: model
title: English bert_base_spanish_wwm_cased_finetuned_nlp_ie_pipeline pipeline BertForSequenceClassification from Willy
author: John Snow Labs
name: bert_base_spanish_wwm_cased_finetuned_nlp_ie_pipeline
date: 2025-04-05
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

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bert_base_spanish_wwm_cased_finetuned_nlp_ie_pipeline` is a English model originally trained by Willy.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_base_spanish_wwm_cased_finetuned_nlp_ie_pipeline_en_5.5.1_3.0_1743869892692.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_base_spanish_wwm_cased_finetuned_nlp_ie_pipeline_en_5.5.1_3.0_1743869892692.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("bert_base_spanish_wwm_cased_finetuned_nlp_ie_pipeline", lang = "en")
annotations =  pipeline.transform(df)
```
```scala
val pipeline = new PretrainedPipeline("bert_base_spanish_wwm_cased_finetuned_nlp_ie_pipeline", lang = "en")
val annotations = pipeline.transform(df)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_base_spanish_wwm_cased_finetuned_nlp_ie_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|411.7 MB|

## References

References

https://huggingface.co/Willy/bert-base-spanish-wwm-cased-finetuned-NLP-IE

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification