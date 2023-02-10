---
layout: model
title: Arabic BertForSequenceClassification Base Cased model (from CAMeL-Lab)
author: John Snow Labs
name: bert_classifier_bert_base_arabic_camelbert_mix_poetry
date: 2022-09-06
tags: [ar, open_source, bert, sequence_classification, classification]
task: Text Classification
language: ar
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
annotator: BertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `bert-base-arabic-camelbert-mix-poetry` is a Arabic model originally trained by `CAMeL-Lab`.

## Predicted Entities

`المواليا`, `الهزج`, `المجتث`, `الدوبيت`, `البسيط`, `المتدارك`, `شعر حر`, `الرجز`, `المنسرح`, `الكامل`, `المديد`, `المقتضب`, `الوافر`, `السريع`, `المتقارب`, `الرمل`, `عامي`, `الخفيف`, `شعر التفعيلة`, `المضارع`, `الطويل`, `السلسلة`, `موشح`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_classifier_bert_base_arabic_camelbert_mix_poetry_ar_4.1.0_3.0_1662507485368.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_classifier_bert_base_arabic_camelbert_mix_poetry_ar_4.1.0_3.0_1662507485368.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

seq_classifier = BertForSequenceClassification.pretrained("bert_classifier_bert_base_arabic_camelbert_mix_poetry","ar") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("class")
    
pipeline = Pipeline(stages=[documentAssembler, tokenizer, seq_classifier])

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
 
val seq_classifier = BertForSequenceClassification.pretrained("bert_classifier_bert_base_arabic_camelbert_mix_poetry","ar") 
    .setInputCols(Array("document", "token")) 
    .setOutputCol("class")
   
val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, seq_classifier))

val data = Seq("PUT YOUR STRING HERE").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_classifier_bert_base_arabic_camelbert_mix_poetry|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|ar|
|Size:|409.5 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

- https://huggingface.co/CAMeL-Lab/bert-base-arabic-camelbert-mix-poetry
- https://arxiv.org/pdf/1905.05700.pdf
- https://arxiv.org/abs/2103.06678
- https://github.com/CAMeL-Lab/CAMeLBERT
- https://github.com/CAMeL-Lab/camel_tools