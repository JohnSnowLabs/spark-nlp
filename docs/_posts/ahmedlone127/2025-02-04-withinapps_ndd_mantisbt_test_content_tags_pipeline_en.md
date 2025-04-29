---
layout: model
title: English withinapps_ndd_mantisbt_test_content_tags_pipeline pipeline DistilBertForSequenceClassification from lgk03
author: John Snow Labs
name: withinapps_ndd_mantisbt_test_content_tags_pipeline
date: 2025-02-04
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

Pretrained DistilBertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`withinapps_ndd_mantisbt_test_content_tags_pipeline` is a English model originally trained by lgk03.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/withinapps_ndd_mantisbt_test_content_tags_pipeline_en_5.5.1_3.0_1738670613264.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/withinapps_ndd_mantisbt_test_content_tags_pipeline_en_5.5.1_3.0_1738670613264.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("withinapps_ndd_mantisbt_test_content_tags_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("withinapps_ndd_mantisbt_test_content_tags_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|withinapps_ndd_mantisbt_test_content_tags_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|249.5 MB|

## References

https://huggingface.co/lgk03/WITHINAPPS_NDD-mantisbt_test-content_tags

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertForSequenceClassification