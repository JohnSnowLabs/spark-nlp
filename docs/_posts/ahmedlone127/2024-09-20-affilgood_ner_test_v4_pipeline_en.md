---
layout: model
title: English affilgood_ner_test_v4_pipeline pipeline RoBertaForTokenClassification from nicolauduran45
author: John Snow Labs
name: affilgood_ner_test_v4_pipeline
date: 2024-09-20
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

Pretrained RoBertaForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`affilgood_ner_test_v4_pipeline` is a English model originally trained by nicolauduran45.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/affilgood_ner_test_v4_pipeline_en_5.5.0_3.0_1726846882261.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/affilgood_ner_test_v4_pipeline_en_5.5.0_3.0_1726846882261.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("affilgood_ner_test_v4_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("affilgood_ner_test_v4_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|affilgood_ner_test_v4_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|466.1 MB|

## References

https://huggingface.co/nicolauduran45/affilgood-ner-test-v4

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForTokenClassification