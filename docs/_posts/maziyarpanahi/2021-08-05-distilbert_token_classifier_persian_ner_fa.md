---
layout: model
title: DistilBERT Token Classification - DistilbertNER for Persian Language Understanding (distilbert_token_classifier_persian_ner)
author: John Snow Labs
name: distilbert_token_classifier_persian_ner
date: 2021-08-05
tags: [fa, farsi, persian, distilbert, open_source, ner, token_classification]
task: Named Entity Recognition
language: fa
edition: Spark NLP 3.2.0
spark_version: 2.4
supported: true
annotator: DistilBertForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

## DistilbertNER

This model fine-tuned for the Named Entity Recognition (NER) task on a mixed NER dataset collected from [ARMAN](https://github.com/HaniehP/PersianNER), [PEYMA](http://nsurl.org/2019-2/tasks/task-7-named-entity-recognition-ner-for-farsi/), and [WikiANN](https://elisa-ie.github.io/wikiann/) that covered ten types of entities: 

- Date (DAT)
- Event (EVE)
- Facility (FAC)
- Location (LOC)
- Money (MON)
- Organization (ORG)
- Percent (PCT)
- Person (PER)
- Product (PRO)
- Time (TIM)

### Dataset Information

|       |   Records |   B-DAT |   B-EVE |   B-FAC |   B-LOC |   B-MON |   B-ORG |   B-PCT |   B-PER |   B-PRO |   B-TIM |   I-DAT |   I-EVE |   I-FAC |   I-LOC |   I-MON |   I-ORG |   I-PCT |   I-PER |   I-PRO |   I-TIM |
|:------|----------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|
| Train |     29133 |    1423 |    1487 |    1400 |   13919 |     417 |   15926 |     355 |   12347 |    1855 |     150 |    1947 |    5018 |    2421 |    4118 |    1059 |   19579 |     573 |    7699 |    1914 |     332 |
| Valid |      5142 |     267 |     253 |     250 |    2362 |     100 |    2651 |      64 |    2173 |     317 |      19 |     373 |     799 |     387 |     717 |     270 |    3260 |     101 |    1382 |     303 |      35 |
| Test  |      6049 |     407 |     256 |     248 |    2886 |      98 |    3216 |      94 |    2646 |     318 |      43 |     568 |     888 |     408 |     858 |     263 |    3967 |     141 |    1707 |     296 |      78 |

## Predicted Entities

- B-DAT
- B-EVE
- B-FAC
- B-LOC
- B-MON
- B-ORG
- B-PCT
- B-PER
- B-PRO
- B-TIM
- I-DAT
- I-EVE
- I-FAC
- I-LOC
- I-MON
- I-ORG
- I-PCT
- I-PER
- I-PRO
- I-TIM
- O

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_token_classifier_persian_ner_fa_3.2.0_2.4_1628188022552.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distilbert_token_classifier_persian_ner_fa_3.2.0_2.4_1628188022552.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

tokenClassifier = DistilBertForTokenClassification \
      .pretrained('distilbert_token_classifier_persian_ner', 'fa') \
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

example = spark.createDataFrame([["این سریال به صورت رسمی در تاریخ دهم می ۲۰۱۱ توسط شبکه فاکس برای پخش رزرو شد."]]).toDF("text")
result = pipeline.fit(example).transform(example)
```
```scala
val document_assembler = DocumentAssembler() 
    .setInputCol("text") 
    .setOutputCol("document")

val tokenizer = Tokenizer() 
    .setInputCols("document") 
    .setOutputCol("token")

val tokenClassifier = DistilBertForTokenClassification.pretrained("distilbert_token_classifier_persian_ner", "fa")
      .setInputCols("document", "token")
      .setOutputCol("ner")
      .setCaseSensitive(false)
      .setMaxSentenceLength(512)

// since output column is IOB/IOB2 style, NerConverter can extract entities
val ner_converter = NerConverter() 
    .setInputCols("document", "token", "ner") 
    .setOutputCol("entities")

val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, tokenClassifier, ner_converter))

val example = Seq.empty["این سریال به صورت رسمی در تاریخ دهم می ۲۰۱۱ توسط شبکه فاکس برای پخش رزرو شد."].toDS.toDF("text")

val result = pipeline.fit(example).transform(example)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_token_classifier_persian_ner|
|Compatibility:|Spark NLP 3.2.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, document]|
|Output Labels:|[ner]|
|Language:|fa|
|Case sensitive:|false|
|Max sentense length:|512|

## Data Source

[https://huggingface.co/HooshvareLab/distilbert-fa-zwnj-base-ner](https://huggingface.co/HooshvareLab/distilbert-fa-zwnj-base-ner)

## Benchmarking

```bash
The following tables summarize the scores obtained by model overall and per each class.

**Overall**

|    Model   | accuracy | precision |  recall  |    f1    |
|:----------:|:--------:|:---------:|:--------:|:--------:|
| Distilbert | 0.994534 |  0.946326 |  0.95504 | 0.950663 |

**Per entities**

|     	| number 	| precision 	|  recall  	|    f1    	|
|:---:	|:------:	|:---------:	|:--------:	|:--------:	|
| DAT 	|   407  	|  0.812048 	| 0.828010 	| 0.819951 	|
| EVE 	|   256  	|  0.955056 	| 0.996094 	| 0.975143 	|
| FAC 	|   248  	|  0.972549 	| 1.000000 	| 0.986083 	|
| LOC 	|  2884  	|  0.968403 	| 0.967060 	| 0.967731 	|
| MON 	|   98   	|  0.925532 	| 0.887755 	| 0.906250 	|
| ORG 	|  3216  	|  0.932095 	| 0.951803 	| 0.941846 	|
| PCT 	|   94   	|  0.936842 	| 0.946809 	| 0.941799 	|
| PER 	|  2645  	|  0.959818 	| 0.957278 	| 0.958546 	|
| PRO 	|   318  	|  0.963526 	| 0.996855 	| 0.979907 	|
| TIM 	|   43   	|  0.760870 	| 0.813953 	| 0.786517 	|

```