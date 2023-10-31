---
layout: model
title: Arabic BertForSequenceClassification Base Cased model (from Abdelrahman-Rezk)
author: John Snow Labs
name: bert_classifier_base_arabic_camel_mix_poetry_finetuned_qawaf
date: 2023-10-31
tags: [bert, sequence_classification, classification, open_source, ar, onnx]
task: Text Classification
language: ar
edition: Spark NLP 5.1.4
spark_version: 3.4
supported: true
engine: onnx
annotator: BertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `bert-base-arabic-camelbert-mix-poetry-finetuned-qawaf` is a Arabic model originally trained by `Abdelrahman-Rezk`.

## Predicted Entities

`المضارع`, `المتقارب`, `المقتضب`, `الهزج`, `السلسلة`, `المجتث`, `الطويل`, `عامي`, `الرمل`, `الرجز`, `الوافر`, `المتدارك`, `المواليا`, `الدوبيت`, `الخفيف`, `شعر التفعيلة`, `المديد`, `السريع`, `المنسرح`, `شعر حر`, `البسيط`, `الكامل`, `موشح`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_classifier_base_arabic_camel_mix_poetry_finetuned_qawaf_ar_5.1.4_3.4_1698794306775.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_classifier_base_arabic_camel_mix_poetry_finetuned_qawaf_ar_5.1.4_3.4_1698794306775.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

sequenceClassifier_loaded = BertForSequenceClassification.pretrained("bert_classifier_base_arabic_camel_mix_poetry_finetuned_qawaf","ar") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("class")

pipeline = Pipeline(stages=[documentAssembler, tokenizer,sequenceClassifier_loaded])

data = spark.createDataFrame([["أنا أحب الشرارة NLP"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
          .setInputCol("text") 
          .setOutputCol("document")

val tokenizer = new Tokenizer() 
    .setInputCols(Array("document"))
    .setOutputCol("token")

val sequenceClassifier_loaded = BertForSequenceClassification.pretrained("bert_classifier_base_arabic_camel_mix_poetry_finetuned_qawaf","ar") 
    .setInputCols(Array("document", "token")) 
    .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer,sequenceClassifier_loaded))

val data = Seq("أنا أحب الشرارة NLP").toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_classifier_base_arabic_camel_mix_poetry_finetuned_qawaf|
|Compatibility:|Spark NLP 5.1.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|ar|
|Size:|408.9 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

References

- https://huggingface.co/Abdelrahman-Rezk/bert-base-arabic-camelbert-mix-poetry-finetuned-qawaf