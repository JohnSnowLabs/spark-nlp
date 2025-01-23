---
layout: model
title: English cs221_deberta_v3_large_finetuned_semeval_pipeline pipeline DeBertaForSequenceClassification from Kuongan
author: John Snow Labs
name: cs221_deberta_v3_large_finetuned_semeval_pipeline
date: 2025-01-23
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

Pretrained DeBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`cs221_deberta_v3_large_finetuned_semeval_pipeline` is a English model originally trained by Kuongan.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/cs221_deberta_v3_large_finetuned_semeval_pipeline_en_5.5.1_3.0_1737658778126.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/cs221_deberta_v3_large_finetuned_semeval_pipeline_en_5.5.1_3.0_1737658778126.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("cs221_deberta_v3_large_finetuned_semeval_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("cs221_deberta_v3_large_finetuned_semeval_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|cs221_deberta_v3_large_finetuned_semeval_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.5 GB|

## References

https://huggingface.co/Kuongan/CS221-deberta-v3-large-finetuned-semeval

## Included Models

- DocumentAssembler
- TokenizerModel
- DeBertaForSequenceClassification