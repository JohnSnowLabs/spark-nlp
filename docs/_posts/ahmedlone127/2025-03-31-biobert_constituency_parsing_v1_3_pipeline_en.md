---
layout: model
title: English biobert_constituency_parsing_v1_3_pipeline pipeline BertForTokenClassification from almo762
author: John Snow Labs
name: biobert_constituency_parsing_v1_3_pipeline
date: 2025-03-31
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

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`biobert_constituency_parsing_v1_3_pipeline` is a English model originally trained by almo762.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/biobert_constituency_parsing_v1_3_pipeline_en_5.5.1_3.0_1743460492500.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/biobert_constituency_parsing_v1_3_pipeline_en_5.5.1_3.0_1743460492500.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("biobert_constituency_parsing_v1_3_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("biobert_constituency_parsing_v1_3_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|biobert_constituency_parsing_v1_3_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|405.7 MB|

## References

https://huggingface.co/almo762/biobert-constituency-parsing-v1.3

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification