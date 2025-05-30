---
layout: model
title: Vietnamese BertForTokenClassification Base Cased model (from KoichiYasuoka)
author: John Snow Labs
name: bert_token_classifier_base_vietnamese_upos
date: 2024-09-06
tags: [vi, open_source, bert, token_classification, ner, onnx]
task: Named Entity Recognition
language: vi
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
engine: onnx
annotator: BertForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `bert-base-vietnamese-upos` is a Vietnamese model originally trained by `KoichiYasuoka`.

## Predicted Entities

`NOUN`, `INTJ`, `AUX`, `ADP`, `DET`, `X`, `SYM`, `NUM`, `PUNCT`, `PRON`, `PROPN`, `VERB`, `ADJ`, `PART`, `CCONJ`, `ADV`, `SCONJ`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_token_classifier_base_vietnamese_upos_vi_5.5.0_3.0_1725663503169.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_token_classifier_base_vietnamese_upos_vi_5.5.0_3.0_1725663503169.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols("document") \
    .setOutputCol("token")

tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_base_vietnamese_upos","vi") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("ner")

pipeline = Pipeline(stages=[documentAssembler, tokenizer, tokenClassifier])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")

val tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_base_vietnamese_upos","vi")
    .setInputCols(Array("document", "token"))
    .setOutputCol("ner")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, tokenClassifier))

val data = Seq("PUT YOUR STRING HERE").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_base_vietnamese_upos|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|vi|
|Size:|429.0 MB|

## References

References

- https://huggingface.co/KoichiYasuoka/bert-base-vietnamese-upos
- https://universaldependencies.org/u/pos/
- https://github.com/KoichiYasuoka/esupar