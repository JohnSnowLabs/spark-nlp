---
layout: model
title: Turkish xlmroberta_ner_xlm_roberta_base_turkish_ner_pipeline pipeline XlmRoBertaForTokenClassification from akdeniz27
author: John Snow Labs
name: xlmroberta_ner_xlm_roberta_base_turkish_ner_pipeline
date: 2024-09-01
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

Pretrained XlmRoBertaForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`xlmroberta_ner_xlm_roberta_base_turkish_ner_pipeline` is a Turkish model originally trained by akdeniz27.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlmroberta_ner_xlm_roberta_base_turkish_ner_pipeline_tr_5.4.2_3.0_1725152300175.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xlmroberta_ner_xlm_roberta_base_turkish_ner_pipeline_tr_5.4.2_3.0_1725152300175.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("xlmroberta_ner_xlm_roberta_base_turkish_ner_pipeline", lang = "tr")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("xlmroberta_ner_xlm_roberta_base_turkish_ner_pipeline", lang = "tr")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlmroberta_ner_xlm_roberta_base_turkish_ner_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|tr|
|Size:|851.0 MB|

## References

https://huggingface.co/akdeniz27/xlm-roberta-base-turkish-ner

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForTokenClassification