---
layout: model
title: English BertForSequenceClassification Base Uncased model (from Hate-speech-CNERG)
author: John Snow Labs
name: bert_classifier_bert_base_uncased_hatexplain_rationale_two
date: 2022-09-07
tags: [en, open_source, bert, sequence_classification, classification]
task: Text Classification
language: en
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
annotator: BertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `bert-base-uncased-hatexplain-rationale-two` is a English model originally trained by `Hate-speech-CNERG`.

## Predicted Entities

`NORMAL`, `ABUSIVE`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_classifier_bert_base_uncased_hatexplain_rationale_two_en_4.1.0_3.0_1662509264306.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

seq_classifier = BertForSequenceClassification.pretrained("bert_classifier_bert_base_uncased_hatexplain_rationale_two","en") \
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
 
val seq_classifier = BertForSequenceClassification.pretrained("bert_classifier_bert_base_uncased_hatexplain_rationale_two","en") 
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
|Model Name:|bert_classifier_bert_base_uncased_hatexplain_rationale_two|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|410.1 MB|
|Case sensitive:|false|
|Max sentence length:|256|

## References

- https://huggingface.co/Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two
- https://arxiv.org/abs/2012.10289
- https://github.com/punyajoy/HateXplain
- https://aclanthology.org/2021.acl-long.330.pdf
- https://dl.acm.org/doi/pdf/10.1145/3442188.3445922
- https://github.com/hate-alert/HateXplain/tree/master/Preprocess
- https://arxiv.org/pdf/2012.10289.pdf