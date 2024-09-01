---
layout: model
title: Turkish deprem_ner_mdebertav3_pipeline pipeline DeBertaForTokenClassification from deprem-ml
author: John Snow Labs
name: deprem_ner_mdebertav3_pipeline
date: 2024-08-31
tags: [tr, open_source, pipeline, onnx]
task: Named Entity Recognition
language: tr
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DeBertaForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`deprem_ner_mdebertav3_pipeline` is a Turkish model originally trained by deprem-ml.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/deprem_ner_mdebertav3_pipeline_tr_5.4.2_3.0_1725114570573.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/deprem_ner_mdebertav3_pipeline_tr_5.4.2_3.0_1725114570573.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("deprem_ner_mdebertav3_pipeline", lang = "tr")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("deprem_ner_mdebertav3_pipeline", lang = "tr")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|deprem_ner_mdebertav3_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|tr|
|Size:|837.2 MB|

## References

https://huggingface.co/deprem-ml/deprem-ner-mdebertav3

## Included Models

- DocumentAssembler
- TokenizerModel
- DeBertaForTokenClassification