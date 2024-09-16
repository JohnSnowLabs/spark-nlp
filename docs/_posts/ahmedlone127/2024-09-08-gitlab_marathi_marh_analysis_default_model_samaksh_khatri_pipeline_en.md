---
layout: model
title: English gitlab_marathi_marh_analysis_default_model_samaksh_khatri_pipeline pipeline DistilBertForSequenceClassification from samaksh-khatri
author: John Snow Labs
name: gitlab_marathi_marh_analysis_default_model_samaksh_khatri_pipeline
date: 2024-09-08
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

Pretrained DistilBertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`gitlab_marathi_marh_analysis_default_model_samaksh_khatri_pipeline` is a English model originally trained by samaksh-khatri.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/gitlab_marathi_marh_analysis_default_model_samaksh_khatri_pipeline_en_5.5.0_3.0_1725764579147.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/gitlab_marathi_marh_analysis_default_model_samaksh_khatri_pipeline_en_5.5.0_3.0_1725764579147.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("gitlab_marathi_marh_analysis_default_model_samaksh_khatri_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("gitlab_marathi_marh_analysis_default_model_samaksh_khatri_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|gitlab_marathi_marh_analysis_default_model_samaksh_khatri_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|249.4 MB|

## References

https://huggingface.co/samaksh-khatri/gitlab-mr-analysis-default-model

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertForSequenceClassification