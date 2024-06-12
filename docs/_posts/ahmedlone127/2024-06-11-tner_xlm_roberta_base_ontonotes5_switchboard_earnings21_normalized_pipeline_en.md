---
layout: model
title: English tner_xlm_roberta_base_ontonotes5_switchboard_earnings21_normalized_pipeline pipeline XlmRoBertaForTokenClassification from anonymoussubmissions
author: John Snow Labs
name: tner_xlm_roberta_base_ontonotes5_switchboard_earnings21_normalized_pipeline
date: 2024-06-11
tags: [en, open_source, pipeline, onnx]
task: Named Entity Recognition
language: en
edition: Spark NLP 5.4.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`tner_xlm_roberta_base_ontonotes5_switchboard_earnings21_normalized_pipeline` is a English model originally trained by anonymoussubmissions.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/tner_xlm_roberta_base_ontonotes5_switchboard_earnings21_normalized_pipeline_en_5.4.0_3.0_1718116969169.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/tner_xlm_roberta_base_ontonotes5_switchboard_earnings21_normalized_pipeline_en_5.4.0_3.0_1718116969169.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("tner_xlm_roberta_base_ontonotes5_switchboard_earnings21_normalized_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("tner_xlm_roberta_base_ontonotes5_switchboard_earnings21_normalized_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|tner_xlm_roberta_base_ontonotes5_switchboard_earnings21_normalized_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|859.4 MB|

## References

https://huggingface.co/anonymoussubmissions/tner-xlm-roberta-base-ontonotes5-switchboard-earnings21-normalized

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForTokenClassification