---
layout: model
title: English nasa_impact_nasa_smd_ibm_southern_sotho_v2_classification_finetuned_pipeline pipeline RoBertaForSequenceClassification from arminmehrabian
author: John Snow Labs
name: nasa_impact_nasa_smd_ibm_southern_sotho_v2_classification_finetuned_pipeline
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

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`nasa_impact_nasa_smd_ibm_southern_sotho_v2_classification_finetuned_pipeline` is a English model originally trained by arminmehrabian.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/nasa_impact_nasa_smd_ibm_southern_sotho_v2_classification_finetuned_pipeline_en_5.5.0_3.0_1725613683975.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/nasa_impact_nasa_smd_ibm_southern_sotho_v2_classification_finetuned_pipeline_en_5.5.0_3.0_1725613683975.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("nasa_impact_nasa_smd_ibm_southern_sotho_v2_classification_finetuned_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("nasa_impact_nasa_smd_ibm_southern_sotho_v2_classification_finetuned_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|nasa_impact_nasa_smd_ibm_southern_sotho_v2_classification_finetuned_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|434.5 MB|

## References

https://huggingface.co/arminmehrabian/nasa-impact-nasa-smd-ibm-st-v2-classification-finetuned

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification