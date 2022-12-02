---
layout: model
title: Chinese BertForSequenceClassification Base Cased model (from uer)
author: John Snow Labs
name: bert_classifier_roberta_base_finetuned_jd_binary_chinese
date: 2022-09-07
tags: [zh, open_source, bert, sequence_classification, classification]
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

Pretrained BertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `roberta-base-finetuned-jd-binary-chinese` is a Chinese model originally trained by `uer`.

## Predicted Entities

`positive (stars 4 and 5)`, `negative (stars 1, 2 and 3)`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_classifier_roberta_base_finetuned_jd_binary_chinese_zh_4.1.0_3.0_1662514996779.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

seq_classifier = BertForSequenceClassification.pretrained("bert_classifier_roberta_base_finetuned_jd_binary_chinese","zh") \
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
 
val seq_classifier = BertForSequenceClassification.pretrained("bert_classifier_roberta_base_finetuned_jd_binary_chinese","zh") 
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
|Model Name:|bert_classifier_roberta_base_finetuned_jd_binary_chinese|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|zh|
|Size:|383.6 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

- https://huggingface.co/uer/roberta-base-finetuned-jd-binary-chinese
- https://arxiv.org/abs/1909.05658
- https://github.com/dbiir/UER-py/wiki/Modelzoo
- https://github.com/zhangxiangxiao/glyph
- https://arxiv.org/abs/1708.02657