---
layout: model
title: English persuasive_pairs_deberta_flipped_newdata_pipeline pipeline DeBertaForSequenceClassification from laurabraad
author: John Snow Labs
name: persuasive_pairs_deberta_flipped_newdata_pipeline
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

Pretrained DeBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`persuasive_pairs_deberta_flipped_newdata_pipeline` is a English model originally trained by laurabraad.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/persuasive_pairs_deberta_flipped_newdata_pipeline_en_5.5.1_3.0_1743129467942.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/persuasive_pairs_deberta_flipped_newdata_pipeline_en_5.5.1_3.0_1743129467942.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("persuasive_pairs_deberta_flipped_newdata_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("persuasive_pairs_deberta_flipped_newdata_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|persuasive_pairs_deberta_flipped_newdata_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.5 GB|

## References

https://huggingface.co/laurabraad/persuasive_pairs_deberta_flipped_newData

## Included Models

- DocumentAssembler
- TokenizerModel
- DeBertaForSequenceClassification