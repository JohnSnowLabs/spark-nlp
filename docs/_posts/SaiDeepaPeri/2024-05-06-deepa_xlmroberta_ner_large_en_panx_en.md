---
layout: model
title: Deepa Panx Model for English
author: SaiDeepaPeri
name: deepa_xlmroberta_ner_large_en_panx
date: 2024-05-06
tags: [en, open_source]
task: Named Entity Recognition
language: en
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: false
annotator: XlmRoBertaForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Named Entity Recognition trained on English panx

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/community.johnsnowlabs.com/SaiDeepaPeri/deepa_xlmroberta_ner_large_en_panx_en_4.1.0_3.0_1715017572119.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://community.johnsnowlabs.com/SaiDeepaPeri/deepa_xlmroberta_ner_large_en_panx_en_4.1.0_3.0_1715017572119.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")

token_classifier = XlmRoBertaForTokenClassification.pretrained("deepa_xlmroberta_ner_large_en_panx", "en") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("ner")

ner_converter = NerConverter() \
    .setInputCols(["document", "token", "ner"]) \
    .setOutputCol("ner_chunk")

pipeline = Pipeline(stages=[documentAssembler, tokenizer, token_classifier, ner_converter])

```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|deepa_xlmroberta_ner_large_en_panx|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Community|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|1.8 GB|
|Case sensitive:|true|
|Max sentence length:|256|