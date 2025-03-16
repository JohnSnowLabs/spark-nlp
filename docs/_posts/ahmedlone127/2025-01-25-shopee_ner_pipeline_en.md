---
layout: model
title: English shopee_ner_pipeline pipeline DistilBertForTokenClassification from yzzhu
author: John Snow Labs
name: shopee_ner_pipeline
date: 2025-01-25
tags: [en, open_source, pipeline, onnx]
task: Named Entity Recognition
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

Pretrained DistilBertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`shopee_ner_pipeline` is a English model originally trained by yzzhu.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/shopee_ner_pipeline_en_5.5.1_3.0_1737833126353.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/shopee_ner_pipeline_en_5.5.1_3.0_1737833126353.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("shopee_ner_pipeline", lang = "en")
annotations =  pipeline.transform(df)
```
```scala
val pipeline = new PretrainedPipeline("shopee_ner_pipeline", lang = "en")
val annotations = pipeline.transform(df)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|shopee_ner_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|820.7 MB|

## References

References

https://huggingface.co/yzzhu/shopee-ner

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForTokenClassification