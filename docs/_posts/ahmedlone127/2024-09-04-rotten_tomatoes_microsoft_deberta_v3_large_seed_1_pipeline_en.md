---
layout: model
title: English rotten_tomatoes_microsoft_deberta_v3_large_seed_1_pipeline pipeline DeBertaForSequenceClassification from utahnlp
author: John Snow Labs
name: rotten_tomatoes_microsoft_deberta_v3_large_seed_1_pipeline
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

Pretrained DeBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`rotten_tomatoes_microsoft_deberta_v3_large_seed_1_pipeline` is a English model originally trained by utahnlp.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/rotten_tomatoes_microsoft_deberta_v3_large_seed_1_pipeline_en_5.5.0_3.0_1725467817730.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/rotten_tomatoes_microsoft_deberta_v3_large_seed_1_pipeline_en_5.5.0_3.0_1725467817730.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("rotten_tomatoes_microsoft_deberta_v3_large_seed_1_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("rotten_tomatoes_microsoft_deberta_v3_large_seed_1_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|rotten_tomatoes_microsoft_deberta_v3_large_seed_1_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.5 GB|

## References

https://huggingface.co/utahnlp/rotten_tomatoes_microsoft_deberta-v3-large_seed-1

## Included Models

- DocumentAssembler
- TokenizerModel
- DeBertaForSequenceClassification