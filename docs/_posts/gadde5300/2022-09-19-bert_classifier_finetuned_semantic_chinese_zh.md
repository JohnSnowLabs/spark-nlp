---
layout: model
title: Chinese BertForSequenceClassification Cased model (from Ayazhankad)
author: John Snow Labs
name: bert_classifier_finetuned_semantic_chinese
date: 2022-09-19
tags: [bert, sequence_classification, classification, open_source, zh]
task: Text Classification
language: zh
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
annotator: BertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `bert-finetuned-semantic-chinese` is a Chinese model originally trained by `Ayazhankad`.

## Predicted Entities

`Star_1`, `Star_2`, `Star_3`, `Star_4`, `Star_5`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_classifier_finetuned_semantic_chinese_zh_4.1.0_3.0_1663607853128.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_classifier_finetuned_semantic_chinese_zh_4.1.0_3.0_1663607853128.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

sequenceClassifier_loaded = BertForSequenceClassification.pretrained("bert_classifier_finetuned_semantic_chinese","zh") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("class")

pipeline = Pipeline(stages=[documentAssembler, tokenizer,sequenceClassifier_loaded])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
          .setInputCol("text") 
          .setOutputCol("document")

val tokenizer = new Tokenizer() 
    .setInputCols(Array("document"))
    .setOutputCol("token")

val sequenceClassifier_loaded = BertForSequenceClassification.pretrained("bert_classifier_finetuned_semantic_chinese","zh") 
    .setInputCols(Array("document", "token")) 
    .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer,sequenceClassifier_loaded))

val data = Seq("PUT YOUR STRING HERE").toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_classifier_finetuned_semantic_chinese|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|zh|
|Size:|383.8 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

- https://huggingface.co/Ayazhankad/bert-finetuned-semantic-chinese
- https://www.kaggle.com/datasets/utmhikari/doubanmovieshortcomments
- https://www.kaggle.com
- https://en.wikipedia.org/wiki/Douban#:~:text=Douban.com%20(Chinese%3A%20%E8%B1%86%E7%93%A3,and%20activities%20in%20Chinese%20cities.
- https://www.kaggle.com/datasets/utmhikari/doubanmovieshortcomments
- https://www.kaggle.com