---
layout: model
title: Estonian BertForTokenClassification Cased model (from tartuNLP)
author: John Snow Labs
name: bert_token_classifier_est_ner_v2
date: 2023-03-20
tags: [et, open_source, bert, token_classification, ner, tensorflow]
task: Named Entity Recognition
language: et
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

Pretrained BertForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `EstBERT_NER_v2` is a Estonian model originally trained by `tartuNLP`.

## Predicted Entities

`TIME`, `ORG`, `MONEY`, `PER`, `GPE`, `DATE`, `PERCENT`, `TITLE`, `LOC`, `EVENT`, `PROD`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_token_classifier_est_ner_v2_et_4.3.1_3.0_1679332189761.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_token_classifier_est_ner_v2_et_4.3.1_3.0_1679332189761.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
    .setInputCols(["text"]) \
    .setOutputCols("document")

tokenizer = Tokenizer() \
    .setInputCols("document") \
    .setOutputCol("token")

tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_est_ner_v2","et") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("ner")

pipeline = Pipeline(stages=[documentAssembler, tokenizer, tokenClassifier])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
    .setInputCols(Array("text")) 
    .setOutputCols(Array("document"))
      
val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")
 
val tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_est_ner_v2","et") 
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
|Model Name:|bert_token_classifier_est_ner_v2|
|Compatibility:|Spark NLP 4.3.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|et|
|Size:|464.1 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/tartuNLP/EstBERT_NER_v2
- https://metashare.ut.ee/repository/browse/reannotated-estonian-ner-corpus/bd43f1f614a511eca6e4fa163e9d45477d086613d2894fd5af79bf13e3f13594/
- https://metashare.ut.ee/repository/browse/new-estonian-ner-corpus/98b6706c963c11eba6e4fa163e9d45470bcd0533b6994c93ab8b8c628516ffed/