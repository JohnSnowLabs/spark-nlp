---
layout: model
title: Arabic BertForTokenClassification Cased model (from boda)
author: John Snow Labs
name: bert_token_classifier_aner
date: 2023-03-20
tags: [ar, open_source, bert, token_classification, ner, tensorflow]
task: Named Entity Recognition
language: ar
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

Pretrained BertForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `ANER` is a Arabic model originally trained by `boda`.

## Predicted Entities

` أرض  `, ` كتاب  `, ` عتاد  `, ` محامي  `, ` حاد  `, ` رماية  `, ` إعلام  `, ` تعليمي  `, ` أرض طبيعية  `, ` عالم  `, ` قارة  `, ` ولاية أو مقاطعة  `, ` هواء  `, ` فنان  `, ` ترفيه  `, ` نبات  `, ` مركز سكني  `, ` مطار  `, ` علوم طبية  `, ` رياضي  `, ` منشأة منطقة فرعية  `, ` رياضة  `, ` طعام  `, ` مسطح مائي  `, ` مهندس  `, ` شخص  `, ` ماء  `, ` شعب(أمة)  `, ` فظ  `, ` مدينة أو ضاحية  `, ` رجل أعمال  `, ` أراضي البناء  `, ` مجموعة  `, `   `, ` قذيفة  `, ` سياسي  `, ` فيلم  `, ` كيميائي  `, ` سماوي  `, ` سوفتوير(برمجيات)  `, ` تجاري  `, ` صوت  `, ` نووي  `, ` مؤسسة دينية  `, ` حكومة  `, ` شرطة  `, ` دواء  `, ` شخص ديني  `, ` غير حكومي  `, ` منفجر  `, ` طريق  `

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_token_classifier_aner_ar_4.3.1_3.0_1679332803744.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_token_classifier_aner_ar_4.3.1_3.0_1679332803744.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_aner","ar") \
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

val tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_aner","ar")
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
|Model Name:|bert_token_classifier_aner|
|Compatibility:|Spark NLP 4.3.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|ar|
|Size:|505.9 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/boda/ANER
- https://drive.google.com/file/d/1jJn3iWqOeLzaNvO-6aKfgidzJlWOtvti/view?usp=sharing
- https://fsalotaibi.kau.edu.sa/Pages-Arabic-NE-Corpora.aspx
- https://github.com/aub-mind/arabert
- https://fsalotaibi.kau.edu.sa/Pages-Arabic-NE-Corpora.aspx
- https://github.com/BodaSadalla98