---
layout: model
title: Persian Named Entity Recognition (from HooshvareLab)
author: John Snow Labs
name: bert_ner_bert_fa_base_uncased_ner_peyma
date: 2022-05-09
tags: [bert, ner, token_classification, fa, open_source]
task: Named Entity Recognition
language: fa
edition: Spark NLP 3.4.2
spark_version: 3.0
supported: true
annotator: BertForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Named Entity Recognition model, uploaded to Hugging Face, adapted and imported into Spark NLP. `bert-fa-base-uncased-ner-peyma` is a Persian model orginally trained by `HooshvareLab`.

## Predicted Entities

`LOC`, `PER`, `TIM`, `MON`, `DAT`, `PCT`, `ORG`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_ner_bert_fa_base_uncased_ner_peyma_fa_3.4.2_3.0_1652099750115.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
        .setInputCol("text") \
        .setOutputCol("document")

sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")\
       .setInputCols(["document"])\
       .setOutputCol("sentence")

tokenizer = Tokenizer() \
    .setInputCols("sentence") \
    .setOutputCol("token")

tokenClassifier = BertForTokenClassification.pretrained("bert_ner_bert_fa_base_uncased_ner_peyma","fa") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("ner")

pipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, tokenClassifier])

data = spark.createDataFrame([["من عاشق جرقه nlp هستم"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
          .setInputCol("text") 
          .setOutputCol("document")

val sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")
       .setInputCols(Array("document"))
       .setOutputCol("sentence")

val tokenizer = new Tokenizer() 
    .setInputCols(Array("sentence"))
    .setOutputCol("token")

val tokenClassifier = BertForTokenClassification.pretrained("bert_ner_bert_fa_base_uncased_ner_peyma","fa") 
    .setInputCols(Array("sentence", "token")) 
    .setOutputCol("ner")

val pipeline = new Pipeline().setStages(Array(documentAssembler,sentenceDetector, tokenizer, tokenClassifier))

val data = Seq("من عاشق جرقه nlp هستم").toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_ner_bert_fa_base_uncased_ner_peyma|
|Compatibility:|Spark NLP 3.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|fa|
|Size:|607.1 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/HooshvareLab/bert-fa-base-uncased-ner-peyma
- https://github.com/hooshvare/parsbert
- http://nsurl.org/tasks/task-7-named-entity-recognition-ner-for-farsi/
- https://github.com/hooshvare/parsbert-ner/blob/master/persian-ner-pipeline.ipynb
- https://colab.research.google.com/github/hooshvare/parsbert-ner/blob/master/persian-ner-pipeline.ipynb
- https://github.com/hooshvare/parsbert/issues