---
layout: model
title: English finetuned_iitp_pdt_review_roberta_hinglish_big_pipeline pipeline RoBertaForSequenceClassification from aditeyabaral
author: John Snow Labs
name: finetuned_iitp_pdt_review_roberta_hinglish_big_pipeline
date: 2024-09-16
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

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`finetuned_iitp_pdt_review_roberta_hinglish_big_pipeline` is a English model originally trained by aditeyabaral.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/finetuned_iitp_pdt_review_roberta_hinglish_big_pipeline_en_5.5.0_3.0_1726470859007.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/finetuned_iitp_pdt_review_roberta_hinglish_big_pipeline_en_5.5.0_3.0_1726470859007.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("finetuned_iitp_pdt_review_roberta_hinglish_big_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("finetuned_iitp_pdt_review_roberta_hinglish_big_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finetuned_iitp_pdt_review_roberta_hinglish_big_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|251.6 MB|

## References

https://huggingface.co/aditeyabaral/finetuned-iitp_pdt_review-roberta-hinglish-big

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification