---
layout: model
title: English BertForTokenClassification Cased model (from ncats)
author: John Snow Labs
name: bert_ner_EpiExtract4GARD_v2
date: 2022-07-09
tags: [bert, ner, open_source, en]
task: Named Entity Recognition
language: en
edition: Spark NLP 4.0.0
spark_version: 3.0
supported: true
annotator: BertForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `EpiExtract4GARD-v2` is a English model originally trained by `ncats`.

## Predicted Entities

`ETHN`, `LOC`, `SEX`, `DATE`, `STAT`, `EPI`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_ner_EpiExtract4GARD_v2_en_4.0.0_3.0_1657355017060.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
    .setOutputCol("ner")

pipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, tokenClassifier])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

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
    .setOutputCol("ner")

val pipeline = new Pipeline().setStages(Array(documentAssembler,sentenceDetector, tokenizer, tokenClassifier))

val data = Seq("PUT YOUR STRING HERE").toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_ner_EpiExtract4GARD_v2|
|Compatibility:|Spark NLP 4.0.0+|
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
- https://github.com/ncats/epi4GARD/blob/master/EpiExtract4GARD/extract_abs.py
- https://github.com/ncats/epi4GARD/blob/master/EpiExtract4GARD/gard-id-name-synonyms.json
- https://github.com/ncats/epi4GARD/tree/master/EpiExtract4GARD#epiextract4gard
- https://aws.amazon.com/ec2/instance-types/
- https://pubmed.ncbi.nlm.nih.gov/21659675/
- https://github.com/ncats/epi4GARD/blob/master/EpiExtract4GARD/Case%20Study.ipynb
- https://github.com/wzkariampuzha
- https://github.com/ncats/epi4GARD/blob/master/EpiExtract4GARD/classify_abs.py