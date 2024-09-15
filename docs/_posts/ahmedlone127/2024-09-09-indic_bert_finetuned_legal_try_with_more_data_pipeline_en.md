---
layout: model
title: English indic_bert_finetuned_legal_try_with_more_data_pipeline pipeline AlbertForSequenceClassification from PoptropicaSahil
author: John Snow Labs
name: indic_bert_finetuned_legal_try_with_more_data_pipeline
date: 2024-09-09
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

Pretrained AlbertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`indic_bert_finetuned_legal_try_with_more_data_pipeline` is a English model originally trained by PoptropicaSahil.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/indic_bert_finetuned_legal_try_with_more_data_pipeline_en_5.5.0_3.0_1725924817893.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/indic_bert_finetuned_legal_try_with_more_data_pipeline_en_5.5.0_3.0_1725924817893.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("indic_bert_finetuned_legal_try_with_more_data_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("indic_bert_finetuned_legal_try_with_more_data_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|indic_bert_finetuned_legal_try_with_more_data_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|127.8 MB|

## References

https://huggingface.co/PoptropicaSahil/indic-bert-finetuned-legal_try_with_more_data

## Included Models

- DocumentAssembler
- TokenizerModel
- AlbertForSequenceClassification