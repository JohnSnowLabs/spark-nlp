---
layout: model
title: English xlmr_sinhalese_english_all_shuffled_764_test1000_pipeline pipeline XlmRoBertaForSequenceClassification from patpizio
author: John Snow Labs
name: xlmr_sinhalese_english_all_shuffled_764_test1000_pipeline
date: 2025-03-28
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

Pretrained XlmRoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`xlmr_sinhalese_english_all_shuffled_764_test1000_pipeline` is a English model originally trained by patpizio.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlmr_sinhalese_english_all_shuffled_764_test1000_pipeline_en_5.5.1_3.0_1743151908884.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xlmr_sinhalese_english_all_shuffled_764_test1000_pipeline_en_5.5.1_3.0_1743151908884.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("xlmr_sinhalese_english_all_shuffled_764_test1000_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("xlmr_sinhalese_english_all_shuffled_764_test1000_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlmr_sinhalese_english_all_shuffled_764_test1000_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|814.3 MB|

## References

https://huggingface.co/patpizio/xlmr-si-en-all_shuffled-764-test1000

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForSequenceClassification