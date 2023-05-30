---
layout: model
title: English BertForTokenClassification Base Cased model (from Jzuluaga)
author: John Snow Labs
name: bert_token_classifier_base_ner_atc_en_atco2_1h
date: 2023-03-20
tags: [en, open_source, bert, token_classification, ner, tensorflow]
task: Named Entity Recognition
language: en
edition: Spark NLP 4.3.1
spark_version: 3.0
supported: true
engine: tensorflow
annotator: BertForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `bert-base-ner-atc-en-atco2-1h` is a English model originally trained by `Jzuluaga`.

## Predicted Entities

`command`, `value`, `callsign`, `O`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_token_classifier_base_ner_atc_en_atco2_1h_en_4.3.1_3.0_1679332513542.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_token_classifier_base_ner_atc_en_atco2_1h_en_4.3.1_3.0_1679332513542.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_base_ner_atc_en_atco2_1h","en") \
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

val tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_base_ner_atc_en_atco2_1h","en")
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
|Model Name:|bert_token_classifier_base_ner_atc_en_atco2_1h|
|Compatibility:|Spark NLP 4.3.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|407.8 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/Jzuluaga/bert-base-ner-atc-en-atco2-1h
- https://github.com/idiap/atco2-corpus
- https://arxiv.org/abs/2211.04054
- https://www.atco2.org/data
- https://github.com/idiap/atco2-corpus
- https://arxiv.org/abs/2211.04054
- https://www.atco2.org/data
- https://github.com/idiap/atco2-corpus/tree/main/data/databases/atco2_test_set_1h/data_prepare_atco2_corpus_other.sh
- https://paperswithcode.com/sota?task=ner&dataset=ATCO2+corpus+%28Air+Traffic+Control+Communications%29