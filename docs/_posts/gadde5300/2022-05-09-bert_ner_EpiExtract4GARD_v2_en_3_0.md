---
layout: model
title: English Named Entity Recognition (from ncats)
author: John Snow Labs
name: bert_ner_EpiExtract4GARD_v2
date: 2022-05-09
tags: [bert, ner, token_classification, en, open_source]
task: Named Entity Recognition
language: en
edition: Spark NLP 3.4.2
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Named Entity Recognition model, uploaded to Hugging Face, adapted and imported into Spark NLP. `EpiExtract4GARD-v2` is a English model orginally trained by `ncats`.

## Predicted Entities

`LOC`, `DATE`, `STAT`, `ETHN`, `EPI`, `SEX`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_ner_EpiExtract4GARD_v2_en_3.4.2_3.0_1652096877827.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

tokenClassifier = BertForTokenClassification.pretrained("bert_ner_EpiExtract4GARD_v2","en") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("pos")

pipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, tokenClassifier])

data = spark.createDataFrame([["I love Spark NLP"]]).toDF("text")

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

val tokenClassifier = BertForTokenClassification.pretrained("bert_ner_EpiExtract4GARD_v2","en") 
    .setInputCols(Array("sentence", "token")) 
    .setOutputCol("pos")

val pipeline = new Pipeline().setStages(Array(documentAssembler,sentenceDetector, tokenizer, tokenClassifier))

val data = Seq("I love Spark NLP").toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_ner_EpiExtract4GARD_v2|
|Compatibility:|Spark NLP 3.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|1.3 GB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/ncats/EpiExtract4GARD-v2
- https://github.com/ncats/epi4GARD/tree/master/EpiExtract4GARD#epiextract4gard
- https://pubmed.ncbi.nlm.nih.gov/21659675/
- https://github.com/ncats/epi4GARD/blob/master/EpiExtract4GARD/classify_abs.py
- https://github.com/ncats/epi4GARD/blob/master/EpiExtract4GARD/extract_abs.py
- https://github.com/ncats/epi4GARD/blob/master/EpiExtract4GARD/gard-id-name-synonyms.json
- https://github.com/ncats/epi4GARD/blob/master/EpiExtract4GARD/Case%20Study.ipynb
- https://aws.amazon.com/ec2/instance-types/
- https://github.com/wzkariampuzha