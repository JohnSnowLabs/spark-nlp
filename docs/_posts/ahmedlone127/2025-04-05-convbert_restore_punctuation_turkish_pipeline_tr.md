---
layout: model
title: Turkish convbert_restore_punctuation_turkish_pipeline pipeline BertForTokenClassification from uygarkurt
author: John Snow Labs
name: convbert_restore_punctuation_turkish_pipeline
date: 2025-04-05
tags: [tr, open_source, pipeline, onnx]
task: Named Entity Recognition
language: tr
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`convbert_restore_punctuation_turkish_pipeline` is a Turkish model originally trained by uygarkurt.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/convbert_restore_punctuation_turkish_pipeline_tr_5.5.1_3.0_1743851065521.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/convbert_restore_punctuation_turkish_pipeline_tr_5.5.1_3.0_1743851065521.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("convbert_restore_punctuation_turkish_pipeline", lang = "tr")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("convbert_restore_punctuation_turkish_pipeline", lang = "tr")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|convbert_restore_punctuation_turkish_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|tr|
|Size:|400.1 MB|

## References

https://huggingface.co/uygarkurt/convbert-restore-punctuation-turkish

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification