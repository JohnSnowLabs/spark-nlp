---
layout: model
title: English legalro_bert_for_rhetorical_role_labeling_pipeline pipeline RoBertaForSequenceClassification from engineersaloni159
author: John Snow Labs
name: legalro_bert_for_rhetorical_role_labeling_pipeline
date: 2024-09-03
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

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`legalro_bert_for_rhetorical_role_labeling_pipeline` is a English model originally trained by engineersaloni159.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/legalro_bert_for_rhetorical_role_labeling_pipeline_en_5.5.0_3.0_1725403020813.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/legalro_bert_for_rhetorical_role_labeling_pipeline_en_5.5.0_3.0_1725403020813.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("legalro_bert_for_rhetorical_role_labeling_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("legalro_bert_for_rhetorical_role_labeling_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legalro_bert_for_rhetorical_role_labeling_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|468.3 MB|

## References

https://huggingface.co/engineersaloni159/LegalRo-BERt_for_rhetorical_role_labeling

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification