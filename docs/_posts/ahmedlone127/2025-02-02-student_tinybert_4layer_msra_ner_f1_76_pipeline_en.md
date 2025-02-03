---
layout: model
title: English student_tinybert_4layer_msra_ner_f1_76_pipeline pipeline BertForTokenClassification from krisGPT
author: John Snow Labs
name: student_tinybert_4layer_msra_ner_f1_76_pipeline
date: 2025-02-02
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

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`student_tinybert_4layer_msra_ner_f1_76_pipeline` is a English model originally trained by krisGPT.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/student_tinybert_4layer_msra_ner_f1_76_pipeline_en_5.5.1_3.0_1738530372983.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/student_tinybert_4layer_msra_ner_f1_76_pipeline_en_5.5.1_3.0_1738530372983.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("student_tinybert_4layer_msra_ner_f1_76_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("student_tinybert_4layer_msra_ner_f1_76_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|student_tinybert_4layer_msra_ner_f1_76_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|168.6 MB|

## References

https://huggingface.co/krisGPT/student-tinybert-4layer-msra-ner_f1_76

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification