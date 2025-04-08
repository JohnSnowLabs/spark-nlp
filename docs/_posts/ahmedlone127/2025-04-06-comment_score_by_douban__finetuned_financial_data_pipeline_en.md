---
layout: model
title: English comment_score_by_douban__finetuned_financial_data_pipeline pipeline BertForSequenceClassification from lazyfrog
author: John Snow Labs
name: comment_score_by_douban__finetuned_financial_data_pipeline
date: 2025-04-06
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

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`comment_score_by_douban__finetuned_financial_data_pipeline` is a English model originally trained by lazyfrog.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/comment_score_by_douban__finetuned_financial_data_pipeline_en_5.5.1_3.0_1743964487327.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/comment_score_by_douban__finetuned_financial_data_pipeline_en_5.5.1_3.0_1743964487327.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("comment_score_by_douban__finetuned_financial_data_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("comment_score_by_douban__finetuned_financial_data_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|comment_score_by_douban__finetuned_financial_data_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|383.3 MB|

## References

https://huggingface.co/lazyfrog/Comment_Score_By_douban_-finetuned-financial_data

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification