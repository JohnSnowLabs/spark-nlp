---
layout: model
title: Tagalog stud_fac_eval_distilbert_base_uncased_v2_pipeline pipeline DistilBertForSequenceClassification from MENG21
author: John Snow Labs
name: stud_fac_eval_distilbert_base_uncased_v2_pipeline
date: 2025-01-25
tags: [tl, open_source, pipeline, onnx]
task: Text Classification
language: tl
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DistilBertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`stud_fac_eval_distilbert_base_uncased_v2_pipeline` is a Tagalog model originally trained by MENG21.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/stud_fac_eval_distilbert_base_uncased_v2_pipeline_tl_5.5.1_3.0_1737838122650.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/stud_fac_eval_distilbert_base_uncased_v2_pipeline_tl_5.5.1_3.0_1737838122650.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("stud_fac_eval_distilbert_base_uncased_v2_pipeline", lang = "tl")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("stud_fac_eval_distilbert_base_uncased_v2_pipeline", lang = "tl")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|stud_fac_eval_distilbert_base_uncased_v2_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|tl|
|Size:|249.5 MB|

## References

https://huggingface.co/MENG21/stud-fac-eval-distilbert-base-uncased_v2

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertForSequenceClassification