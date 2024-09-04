---
layout: model
title: English xlmr_romanian_english_all_shuffled_1985_test1000_pipeline pipeline XlmRoBertaForSequenceClassification from patpizio
author: John Snow Labs
name: xlmr_romanian_english_all_shuffled_1985_test1000_pipeline
date: 2024-09-04
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

Pretrained XlmRoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`xlmr_romanian_english_all_shuffled_1985_test1000_pipeline` is a English model originally trained by patpizio.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlmr_romanian_english_all_shuffled_1985_test1000_pipeline_en_5.5.0_3.0_1725410776181.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xlmr_romanian_english_all_shuffled_1985_test1000_pipeline_en_5.5.0_3.0_1725410776181.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("xlmr_romanian_english_all_shuffled_1985_test1000_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("xlmr_romanian_english_all_shuffled_1985_test1000_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlmr_romanian_english_all_shuffled_1985_test1000_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|820.3 MB|

## References

https://huggingface.co/patpizio/xlmr-ro-en-all_shuffled-1985-test1000

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForSequenceClassification