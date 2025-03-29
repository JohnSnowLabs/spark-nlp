---
layout: model
title: English vietnamese_sentimental_analysis_pipeline pipeline DistilBertForSequenceClassification from thanhchauns2
author: John Snow Labs
name: vietnamese_sentimental_analysis_pipeline
date: 2025-03-29
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

Pretrained DistilBertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`vietnamese_sentimental_analysis_pipeline` is a English model originally trained by thanhchauns2.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/vietnamese_sentimental_analysis_pipeline_en_5.5.1_3.0_1743265223454.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/vietnamese_sentimental_analysis_pipeline_en_5.5.1_3.0_1743265223454.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("vietnamese_sentimental_analysis_pipeline", lang = "en")
annotations =  pipeline.transform(df)
```
```scala
val pipeline = new PretrainedPipeline("vietnamese_sentimental_analysis_pipeline", lang = "en")
val annotations = pipeline.transform(df)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|vietnamese_sentimental_analysis_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|249.5 MB|

## References

References

https://huggingface.co/thanhchauns2/vietnamese-sentimental-analysis

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertForSequenceClassification