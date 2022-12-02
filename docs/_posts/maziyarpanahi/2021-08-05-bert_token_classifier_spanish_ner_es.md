---
layout: model
title: BERT Token Classification - BETO Spanish Language Understanding (bert_token_classifier_spanish_ner)
author: John Snow Labs
name: bert_token_classifier_spanish_ner
date: 2021-08-05
tags: [spanish, es, bert, ner, token_classification, open_source]
task: Named Entity Recognition
language: es
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

This model is a fine-tuned on [NER-C](https://www.kaggle.com/nltkdata/conll-corpora) version of the Spanish BERT cased [(BETO)](https://github.com/dccuchile/beto) for **NER** downstream task.

## Predicted Entities

- B-LOC
- B-MISC
- B-ORG
- B-PER
- I-LOC
- I-MISC
- I-ORG
- I-PER
- O

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_token_classifier_spanish_ner_es_3.2.0_2.4_1628186970366.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
.pretrained('bert_token_classifier_spanish_ner', 'es') \
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

example = spark.createDataFrame([["Me llamo Wolfgang y vivo en Berlin"]]).toDF("text")
result = pipeline.fit(example).transform(example)
```
```scala
val document_assembler = DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val tokenizer = Tokenizer() 
.setInputCols("document") 
.setOutputCol("token")

val tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_spanish_ner", "es")
.setInputCols("document", "token")
.setOutputCol("ner")
.setCaseSensitive(true)
.setMaxSentenceLength(512)

// since output column is IOB/IOB2 style, NerConverter can extract entities
val ner_converter = NerConverter() 
.setInputCols("document", "token", "ner") 
.setOutputCol("entities")

val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, tokenClassifier, ner_converter))

val example = Seq.empty["Me llamo Wolfgang y vivo en Berlin"].toDS.toDF("text")

val result = pipeline.fit(example).transform(example)
```


{:.nlu-block}
```python
import nlu
nlu.load("es.classify.token_bert.spanish_ner").predict("""Me llamo Wolfgang y vivo en Berlin""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_spanish_ner|
|Compatibility:|Spark NLP 3.2.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, document]|
|Output Labels:|[ner]|
|Language:|es|
|Case sensitive:|false|
|Max sentense length:|512|

## Data Source

[https://huggingface.co/mrm8488/bert-spanish-cased-finetuned-ner](https://huggingface.co/mrm8488/bert-spanish-cased-finetuned-ner)

## Benchmarking

```bash
|                                                      Metric                                                       |  # score  |
| :------------------------------------------------------------------------------------: | :-------: |
| F1                                       | **90.17**  
| Precision                                | **89.86** | 
| Recall                                   | **90.47** |    

## Comparison:

|                                                      Model                                                       |  # F1 score  |Size(MB)|
| :--------------------------------------------------------------------------------------------------------------: | :-------: |:------|
|                                        bert-base-spanish-wwm-cased (BETO)                                        |   88.43   | 421
| [bert-spanish-cased-finetuned-ner (this one)](https://huggingface.co/mrm8488/bert-spanish-cased-finetuned-ner) | **90.17** | 420 |
|                                              Best Multilingual BERT                                              |   87.38   | 681 |
|[TinyBERT-spanish-uncased-finetuned-ner](https://huggingface.co/mrm8488/TinyBERT-spanish-uncased-finetuned-ner) | 70.00 | **55** |

```
