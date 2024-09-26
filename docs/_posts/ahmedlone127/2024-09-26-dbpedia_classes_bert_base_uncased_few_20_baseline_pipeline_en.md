---
layout: model
title: English dbpedia_classes_bert_base_uncased_few_20_baseline_pipeline pipeline BertForSequenceClassification from TheChickenAgent
author: John Snow Labs
name: dbpedia_classes_bert_base_uncased_few_20_baseline_pipeline
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

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`dbpedia_classes_bert_base_uncased_few_20_baseline_pipeline` is a English model originally trained by TheChickenAgent.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/dbpedia_classes_bert_base_uncased_few_20_baseline_pipeline_en_5.5.0_3.0_1727321938787.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/dbpedia_classes_bert_base_uncased_few_20_baseline_pipeline_en_5.5.0_3.0_1727321938787.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("dbpedia_classes_bert_base_uncased_few_20_baseline_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("dbpedia_classes_bert_base_uncased_few_20_baseline_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|dbpedia_classes_bert_base_uncased_few_20_baseline_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|409.4 MB|

## References

https://huggingface.co/TheChickenAgent/DBPedia_Classes_BERT-base-uncased-few-20-baseline

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification