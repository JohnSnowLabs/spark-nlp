---
layout: model
title: English 4_datasets_fake_news_with_balanced_with_raw_data_pipeline pipeline DistilBertForSequenceClassification from littlepinhorse
author: John Snow Labs
name: 4_datasets_fake_news_with_balanced_with_raw_data_pipeline
date: 2024-09-15
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

Pretrained DistilBertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`4_datasets_fake_news_with_balanced_with_raw_data_pipeline` is a English model originally trained by littlepinhorse.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/4_datasets_fake_news_with_balanced_with_raw_data_pipeline_en_5.5.0_3.0_1726385023653.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/4_datasets_fake_news_with_balanced_with_raw_data_pipeline_en_5.5.0_3.0_1726385023653.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("4_datasets_fake_news_with_balanced_with_raw_data_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("4_datasets_fake_news_with_balanced_with_raw_data_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|4_datasets_fake_news_with_balanced_with_raw_data_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|249.5 MB|

## References

https://huggingface.co/littlepinhorse/4_datasets_fake_news_with_Balanced_With_Raw_Data

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertForSequenceClassification