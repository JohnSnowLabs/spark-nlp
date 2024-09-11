---
layout: model
title: None research_paper_mlm_final_label_400k_pipeline pipeline RoBertaForSequenceClassification from ManojAlexender
author: John Snow Labs
name: research_paper_mlm_final_label_400k_pipeline
date: 2024-09-11
tags: [nan, open_source, pipeline, onnx]
task: Text Classification
language: nan
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`research_paper_mlm_final_label_400k_pipeline` is a None model originally trained by ManojAlexender.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/research_paper_mlm_final_label_400k_pipeline_nan_5.5.0_3.0_1726071711500.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/research_paper_mlm_final_label_400k_pipeline_nan_5.5.0_3.0_1726071711500.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("research_paper_mlm_final_label_400k_pipeline", lang = "nan")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("research_paper_mlm_final_label_400k_pipeline", lang = "nan")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|research_paper_mlm_final_label_400k_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|nan|
|Size:|468.6 MB|

## References

https://huggingface.co/ManojAlexender/Research_paper_MLM_Final_Label_400k

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification