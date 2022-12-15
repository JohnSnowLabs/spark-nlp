---
layout: model
title: Turkish BertForTokenClassification Cased model (from alierenak)
author: John Snow Labs
name: bert_token_classifier_berturk_cased_ner
date: 2022-11-30
tags: [tr, open_source, bert, token_classification, ner, tensorflow]
task: Named Entity Recognition
language: tr
edition: Spark NLP 4.2.4
spark_version: 3.0
supported: true
engine: tensorflow
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `berturk-cased-ner` is a Turkish model originally trained by `alierenak`.

## Predicted Entities

`LOCATION`, `ORGANIZATION`, `PERSON`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_token_classifier_berturk_cased_ner_tr_4.2.4_3.0_1669815532606.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_berturk_cased_ner","tr") \
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
 
val tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_berturk_cased_ner","tr") 
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
|Model Name:|bert_token_classifier_berturk_cased_ner|
|Compatibility:|Spark NLP 4.2.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|tr|
|Size:|412.8 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

- https://huggingface.co/alierenak/berturk-cased-ner
- https://www.cambridge.org/core/journals/natural-language-engineering/article/abs/statistical-information-extraction-system-for-turkish/7C288FAFC71D5F0763C1F8CE66464017
- https://aclanthology.org/P11-3019
- https://data.tdd.ai/#/effafb5f-ebfc-4e5c-9a63-4f709ec1a135