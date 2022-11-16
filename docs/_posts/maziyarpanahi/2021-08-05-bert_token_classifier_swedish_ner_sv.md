---
layout: model
title: BERT Token Classification - Swedish Language Understanding (bert_token_classifier_swedish_ner)
author: John Snow Labs
name: bert_token_classifier_swedish_ner
date: 2021-08-05
tags: [swedish, sv, ner, token_classification, open_source, bert]
task: Named Entity Recognition
language: sv
edition: Spark NLP 3.2.0
spark_version: 2.4
supported: true
annotator: BertForTokenClassification
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

`BERT Model` with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks.

[Recorded Future](https://www.recordedfuture.com/) together with [AI Sweden](https://www.ai.se/en) releases a Named Entity Recognition(NER) model for entety detection in Swedish. The model is based on [KB/bert-base-swedish-cased](https://huggingface.co/KB/bert-base-swedish-cased) and finetuned on data collected from various internet sources and forums.

The model has been trained on Swedish data and only supports an inference of Swedish input texts. The model's inference metrics for all non-Swedish inputs are not defined, these inputs are considered as out of domain data.

## Predicted Entities

- Location
- Organization
- Person
- Religion
- Title

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_token_classifier_swedish_ner_sv_3.2.0_2.4_1628187308268.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
.pretrained('bert_token_classifier_swedish_ner', 'sv') \
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

example = spark.createDataFrame([["Engelbert tar Volvon till Tele2 Arena för att titta på Djurgården som spelar fotboll i VM klockan två på kvällen."]]).toDF("text")
result = pipeline.fit(example).transform(example)
```
```scala
val document_assembler = DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val tokenizer = Tokenizer() 
.setInputCols("document") 
.setOutputCol("token")

val tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_swedish_ner", "sv")
.setInputCols("document", "token")
.setOutputCol("ner")
.setCaseSensitive(true)
.setMaxSentenceLength(512)

// since output column is IOB/IOB2 style, NerConverter can extract entities
val ner_converter = NerConverter() 
.setInputCols("document", "token", "ner") 
.setOutputCol("entities")

val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, tokenClassifier, ner_converter))

val example = Seq.empty["Engelbert tar Volvon till Tele2 Arena för att titta på Djurgården som spelar fotboll i VM klockan två på kvällen."].toDS.toDF("text")

val result = pipeline.fit(example).transform(example)
```


{:.nlu-block}
```python
import nlu
nlu.load("sv.classify.token_bert.swedish_ner").predict("""Engelbert tar Volvon till Tele2 Arena för att titta på Djurgården som spelar fotboll i VM klockan två på kvällen.""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_swedish_ner|
|Compatibility:|Spark NLP 3.2.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, document]|
|Output Labels:|[ner]|
|Language:|sv|
|Case sensitive:|false|
|Max sentense length:|512|

## Data Source

[https://huggingface.co/RecordedFuture/Swedish-NER](https://huggingface.co/RecordedFuture/Swedish-NER)

## Benchmarking

```bash
The model had the following metrics when evaluated on test data originating from the same domain as the training data. 

F1-score

| Loc  | Org  | Per  | Nat  | Rel  | Tit  | Total |
|------|------|------|------|------|------|-------|
| 0.91 | 0.88 | 0.96 | 0.95 | 0.91 | 0.84 | 0.92  |
```