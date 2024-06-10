---
layout: model
title: English xlmroberta_sayula_popoluca_xlm_roberta_base_english_upos_pipeline pipeline XlmRoBertaForTokenClassification from KoichiYasuoka
author: John Snow Labs
name: xlmroberta_sayula_popoluca_xlm_roberta_base_english_upos_pipeline
date: 2024-06-10
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

Pretrained XlmRoBertaForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`xlmroberta_sayula_popoluca_xlm_roberta_base_english_upos_pipeline` is a English model originally trained by KoichiYasuoka.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlmroberta_sayula_popoluca_xlm_roberta_base_english_upos_pipeline_en_5.4.0_3.0_1718039745214.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xlmroberta_sayula_popoluca_xlm_roberta_base_english_upos_pipeline_en_5.4.0_3.0_1718039745214.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("xlmroberta_sayula_popoluca_xlm_roberta_base_english_upos_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("xlmroberta_sayula_popoluca_xlm_roberta_base_english_upos_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlmroberta_sayula_popoluca_xlm_roberta_base_english_upos_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|790.5 MB|

## References

https://huggingface.co/KoichiYasuoka/xlm-roberta-base-english-upos

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForTokenClassification