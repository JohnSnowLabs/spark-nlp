---
layout: model
title: English model3e_norwegian_wd_norwegian_perturb_pipeline pipeline DistilBertForTokenClassification from cria111
author: John Snow Labs
name: model3e_norwegian_wd_norwegian_perturb_pipeline
date: 2024-09-07
tags: [en, open_source, pipeline, onnx]
task: Named Entity Recognition
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

Pretrained DistilBertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`model3e_norwegian_wd_norwegian_perturb_pipeline` is a English model originally trained by cria111.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/model3e_norwegian_wd_norwegian_perturb_pipeline_en_5.5.0_3.0_1725739679180.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/model3e_norwegian_wd_norwegian_perturb_pipeline_en_5.5.0_3.0_1725739679180.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("model3e_norwegian_wd_norwegian_perturb_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("model3e_norwegian_wd_norwegian_perturb_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|model3e_norwegian_wd_norwegian_perturb_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|247.3 MB|

## References

https://huggingface.co/cria111/model3e_no_wd_no_perturb

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertForTokenClassification