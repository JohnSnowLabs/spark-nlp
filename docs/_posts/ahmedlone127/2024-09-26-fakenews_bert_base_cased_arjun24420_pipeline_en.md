---
layout: model
title: English fakenews_bert_base_cased_arjun24420_pipeline pipeline BertForSequenceClassification from Arjun24420
author: John Snow Labs
name: fakenews_bert_base_cased_arjun24420_pipeline
date: 2024-09-26
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

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`fakenews_bert_base_cased_arjun24420_pipeline` is a English model originally trained by Arjun24420.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/fakenews_bert_base_cased_arjun24420_pipeline_en_5.5.0_3.0_1727349572603.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/fakenews_bert_base_cased_arjun24420_pipeline_en_5.5.0_3.0_1727349572603.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("fakenews_bert_base_cased_arjun24420_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("fakenews_bert_base_cased_arjun24420_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|fakenews_bert_base_cased_arjun24420_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|405.9 MB|

## References

https://huggingface.co/Arjun24420/FakeNews-BERT-base-cased

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification