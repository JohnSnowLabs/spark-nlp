---
layout: model
title: RoBERTa Token Classification For Persian (roberta_token_classifier_zwnj_base_ner)
author: John Snow Labs
name: roberta_token_classifier_zwnj_base_ner
date: 2021-09-28
tags: [fa, persian, farsi, ner, roberta, token_classification, open_source]
task: Named Entity Recognition
language: fa
edition: Spark NLP 3.3.0
spark_version: 3.0
supported: true
annotator: RoBertaForTokenClassification
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

# RobertaNER

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

## Dataset Information

|       |   Records |   B-DAT |   B-EVE |   B-FAC |   B-LOC |   B-MON |   B-ORG |   B-PCT |   B-PER |   B-PRO |   B-TIM |   I-DAT |   I-EVE |   I-FAC |   I-LOC |   I-MON |   I-ORG |   I-PCT |   I-PER |   I-PRO |   I-TIM |
|:------|----------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|
| Train |     29133 |    1423 |    1487 |    1400 |   13919 |     417 |   15926 |     355 |   12347 |    1855 |     150 |    1947 |    5018 |    2421 |    4118 |    1059 |   19579 |     573 |    7699 |    1914 |     332 |
| Valid |      5142 |     267 |     253 |     250 |    2362 |     100 |    2651 |      64 |    2173 |     317 |      19 |     373 |     799 |     387 |     717 |     270 |    3260 |     101 |    1382 |     303 |      35 |
| Test  |      6049 |     407 |     256 |     248 |    2886 |      98 |    3216 |      94 |    2646 |     318 |      43 |     568 |     888 |     408 |     858 |     263 |    3967 |     141 |    1707 |     296 |      78 |

## Predicted Entities

`DAT`, `EVE`, `FAC`, `LOC`, `MON`, `ORG`, `PCT`, `PER`, `PRO`, `TIM`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_token_classifier_zwnj_base_ner_fa_3.3.0_3.0_1632835301097.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_token_classifier_zwnj_base_ner_fa_3.3.0_3.0_1632835301097.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

tokenClassifier = RoBertaForTokenClassification \
.pretrained('roberta_token_classifier_zwnj_base_ner', 'fa') \
.setInputCols(['token', 'document']) \
.setOutputCol('ner') \
.setCaseSensitive(True) \
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

example = spark.createDataFrame([['در سال ۲۰۱۳ درگذشت و آندرتیکر و کین برای او مراسم یادبود گرفتند.']]).toDF("text")
result = pipeline.fit(example).transform(example)
```
```scala
val document_assembler = DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val tokenizer = Tokenizer() 
.setInputCols("document") 
.setOutputCol("token")

val tokenClassifier = RoBertaForTokenClassification.pretrained("roberta_token_classifier_zwnj_base_ner", "fa")
.setInputCols("document", "token")
.setOutputCol("ner")
.setCaseSensitive(true)
.setMaxSentenceLength(512)

// since output column is IOB/IOB2 style, NerConverter can extract entities
val ner_converter = NerConverter() 
.setInputCols("document", "token", "ner") 
.setOutputCol("entities")

val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, tokenClassifier, ner_converter))

val example = Seq.empty["در سال ۲۰۱۳ درگذشت و آندرتیکر و کین برای او مراسم یادبود گرفتند."].toDS.toDF("text")

val result = pipeline.fit(example).transform(example)
```


{:.nlu-block}
```python
import nlu
nlu.load("fa.classify.token_roberta_token_classifier_zwnj_base_ner").predict("""در سال ۲۰۱۳ درگذشت و آندرتیکر و کین برای او مراسم یادبود گرفتند.""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_token_classifier_zwnj_base_ner|
|Compatibility:|Spark NLP 3.3.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, document]|
|Output Labels:|[ner]|
|Language:|fa|
|Case sensitive:|true|
|Max sentense length:|512|

## Data Source

[https://huggingface.co/HooshvareLab/roberta-fa-zwnj-base-ner](https://huggingface.co/HooshvareLab/roberta-fa-zwnj-base-ner)

## Benchmarking

```bash
## Evaluation

The following tables summarize the scores obtained by model overall and per each class.

**Overall**

|    Model   | accuracy | precision |  recall  |    f1    |
|:----------:|:--------:|:---------:|:--------:|:--------:|
|   Roberta  | 0.994849 |  0.949816 | 0.960235 | 0.954997 |

**Per entities**

|     	| number 	| precision 	|  recall  	|    f1    	|
|:---:	|:------:	|:---------:	|:--------:	|:--------:	|
| DAT 	|   407  	|  0.844869 	| 0.869779 	| 0.857143 	|
| EVE 	|   256  	|  0.948148 	| 1.000000 	| 0.973384 	|
| FAC 	|   248  	|  0.957529 	| 1.000000 	| 0.978304 	|
| LOC 	|  2884  	|  0.965422 	| 0.968100 	| 0.966759 	|
| MON 	|   98   	|  0.937500 	| 0.918367 	| 0.927835 	|
| ORG 	|  3216  	|  0.943662 	| 0.958333 	| 0.950941 	|
| PCT 	|   94   	|  1.000000 	| 0.968085 	| 0.983784 	|
| PER 	|  2646  	|  0.957030 	| 0.959562 	| 0.958294 	|
| PRO 	|   318  	|  0.963636 	| 1.000000 	| 0.981481 	|
| TIM 	|   43   	|  0.739130 	| 0.790698 	| 0.764045 	|

```