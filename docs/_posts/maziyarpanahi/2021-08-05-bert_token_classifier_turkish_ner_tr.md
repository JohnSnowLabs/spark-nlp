---
layout: model
title: BERT Token Classification - Turkish Language Understanding (bert_token_classifier_turkish_ner)
author: John Snow Labs
name: bert_token_classifier_turkish_ner
date: 2021-08-05
tags: [ner, tr, turkish, token_classification, bert, open_source]
task: Named Entity Recognition
language: tr
edition: Spark NLP 3.2.0
spark_version: 2.4
supported: true
annotator: BertForTokenClassification
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Türk Adlandırılmış Varlık Tanıma

**bert_token_classifier_turkish_ner** is a fine-tuned BERT model that is ready to use for **Named Entity Recognition** and achieves **state-of-the-art performance** for the NER task. This model has been trained to recognize four types of entities: location (LOC), organizations (ORG), and person (PER).

## Predicted Entities

- B-LOC
- B-ORG
- B-PER
- I-LOC
- I-ORG
- I-PER
- O

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_token_classifier_turkish_ner_tr_3.2.0_2.4_1628186223192.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_token_classifier_turkish_ner_tr_3.2.0_2.4_1628186223192.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler() \
.setInputCol('text') \
.setOutputCol('document')

tokenizer = Tokenizer() \
.setInputCols(['document']) \
.setOutputCol('token')

tokenClassifier = BertForTokenClassification \
.pretrained('bert_token_classifier_turkish_ner', 'tr') \
.setInputCols(['token', 'document']) \
.setOutputCol('ner') \
.setCaseSensitive(False) \
.setMaxSentenceLength(512)

# since output column is IOB/IOB2 style, NerConverter can extract entities
ner_converter = NerConverter() \
.setInputCols(['document', 'token', 'ner']) \
.setOutputCol('entities')

pipeline = Pipeline(stages=[
document_assembler, 
tokenizer,
tokenClassifier,
ner_converter
])

example = spark.createDataFrame([["İstanbul Türkiye'nin kuzeybatısında, Marmara kıyısı ve Boğaziçi boyunca, Haliç'i de çevreleyecek şekilde kurulmuştur."]]).toDF("text")
result = pipeline.fit(example).transform(example)
```
```scala
val document_assembler = DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val tokenizer = Tokenizer() 
.setInputCols("document") 
.setOutputCol("token")

val tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_turkish_ner", "tr")
.setInputCols("document", "token")
.setOutputCol("ner")
.setCaseSensitive(false)
.setMaxSentenceLength(512)

// since output column is IOB/IOB2 style, NerConverter can extract entities
val ner_converter = NerConverter() 
.setInputCols("document", "token", "ner") 
.setOutputCol("entities")

val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, tokenClassifier, ner_converter))

val example = Seq.empty["İstanbul Türkiye'nin kuzeybatısında, Marmara kıyısı ve Boğaziçi boyunca, Haliç'i de çevreleyecek şekilde kurulmuştur."].toDS.toDF("text")

val result = pipeline.fit(example).transform(example)
```


{:.nlu-block}
```python
import nlu
nlu.load("tr.classify.token_bert.turkish_ner").predict("""İstanbul Türkiye'nin kuzeybatısında, Marmara kıyısı ve Boğaziçi boyunca, Haliç'i de çevreleyecek şekilde kurulmuştur.""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_turkish_ner|
|Compatibility:|Spark NLP 3.2.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, document]|
|Output Labels:|[ner]|
|Language:|tr|
|Case sensitive:|false|
|Max sentense length:|512|

## Data Source

[https://huggingface.co/savasy/bert-base-turkish-ner-cased](https://huggingface.co/savasy/bert-base-turkish-ner-cased)

## Benchmarking

```bash
Eval Results:

* precision = 0.916400580551524
* recall = 0.9342309684101502
* f1 = 0.9252298787412536
* loss = 0.11335893666411284

Test Results:

* precision = 0.9192058759362955
* recall = 0.9303010230367262
* f1 = 0.9247201697271198
* loss = 0.11182546521618497
```