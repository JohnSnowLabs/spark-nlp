---
layout: model
title: English hindi_wordpiece_bert_test_2m_pipeline pipeline BertEmbeddings from rg1683
author: John Snow Labs
name: hindi_wordpiece_bert_test_2m_pipeline
date: 2024-09-22
tags: [en, open_source, pipeline, onnx]
task: Embeddings
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

Pretrained BertEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`hindi_wordpiece_bert_test_2m_pipeline` is a English model originally trained by rg1683.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/hindi_wordpiece_bert_test_2m_pipeline_en_5.5.0_3.0_1727008166350.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/hindi_wordpiece_bert_test_2m_pipeline_en_5.5.0_3.0_1727008166350.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("hindi_wordpiece_bert_test_2m_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("hindi_wordpiece_bert_test_2m_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|hindi_wordpiece_bert_test_2m_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|377.7 MB|

## References

https://huggingface.co/rg1683/hindi_wordpiece_bert_test_2m

## Included Models

- DocumentAssembler
- TokenizerModel
- BertEmbeddings