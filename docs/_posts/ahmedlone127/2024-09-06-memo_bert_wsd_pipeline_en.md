---
layout: model
title: English memo_bert_wsd_pipeline pipeline XlmRoBertaForSequenceClassification from MiMe-MeMo
author: John Snow Labs
name: memo_bert_wsd_pipeline
date: 2024-09-06
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

Pretrained XlmRoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`memo_bert_wsd_pipeline` is a English model originally trained by MiMe-MeMo.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/memo_bert_wsd_pipeline_en_5.5.0_3.0_1725615946286.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/memo_bert_wsd_pipeline_en_5.5.0_3.0_1725615946286.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("memo_bert_wsd_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("memo_bert_wsd_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|memo_bert_wsd_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|466.6 MB|

## References

https://huggingface.co/MiMe-MeMo/MeMo-BERT-WSD

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForSequenceClassification