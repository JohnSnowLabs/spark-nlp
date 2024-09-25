---
layout: model
title: English test_finetuned__roberta_base_biomedical_clinical_spanish__59k_ultrasounds_ner_pipeline pipeline RoBertaForTokenClassification from manucos
author: John Snow Labs
name: test_finetuned__roberta_base_biomedical_clinical_spanish__59k_ultrasounds_ner_pipeline
date: 2024-09-19
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

Pretrained RoBertaForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`test_finetuned__roberta_base_biomedical_clinical_spanish__59k_ultrasounds_ner_pipeline` is a English model originally trained by manucos.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/test_finetuned__roberta_base_biomedical_clinical_spanish__59k_ultrasounds_ner_pipeline_en_5.5.0_3.0_1726730670572.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/test_finetuned__roberta_base_biomedical_clinical_spanish__59k_ultrasounds_ner_pipeline_en_5.5.0_3.0_1726730670572.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("test_finetuned__roberta_base_biomedical_clinical_spanish__59k_ultrasounds_ner_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("test_finetuned__roberta_base_biomedical_clinical_spanish__59k_ultrasounds_ner_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|test_finetuned__roberta_base_biomedical_clinical_spanish__59k_ultrasounds_ner_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|469.5 MB|

## References

https://huggingface.co/manucos/test-finetuned__roberta-base-biomedical-clinical-es__59k-ultrasounds-ner

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForTokenClassification