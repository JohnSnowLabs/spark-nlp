---
layout: model
title: Persian BertForSequenceClassification Base Uncased model (from HooshvareLab)
author: John Snow Labs
name: bert_classifier_bert_fa_base_uncased_sentiment_deepsentipers_binary
date: 2022-09-07
tags: [fa, open_source, bert, sequence_classification, classification]
task: Text Classification
language: fa
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
annotator: BertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `bert-fa-base-uncased-sentiment-deepsentipers-binary` is a Persian model originally trained by `HooshvareLab`.

## Predicted Entities

`negative`, `positive`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_classifier_bert_fa_base_uncased_sentiment_deepsentipers_binary_fa_4.1.0_3.0_1662509753718.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_classifier_bert_fa_base_uncased_sentiment_deepsentipers_binary_fa_4.1.0_3.0_1662509753718.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

seq_classifier = BertForSequenceClassification.pretrained("bert_classifier_bert_fa_base_uncased_sentiment_deepsentipers_binary","fa") \
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
 
val seq_classifier = BertForSequenceClassification.pretrained("bert_classifier_bert_fa_base_uncased_sentiment_deepsentipers_binary","fa") 
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
|Model Name:|bert_classifier_bert_fa_base_uncased_sentiment_deepsentipers_binary|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|fa|
|Size:|609.4 MB|
|Case sensitive:|false|
|Max sentence length:|256|

## References

- https://huggingface.co/HooshvareLab/bert-fa-base-uncased-sentiment-deepsentipers-binary
- https://github.com/hooshvare/parsbert
- https://github.com/phosseini/sentipers
- https://github.com/JoyeBright/DeepSentiPers