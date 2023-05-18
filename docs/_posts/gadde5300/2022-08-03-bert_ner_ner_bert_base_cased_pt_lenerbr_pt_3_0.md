---
layout: model
title: Portugese BertForTokenClassification Base Cased model (from mateusqc)
author: John Snow Labs
name: bert_ner_ner_bert_base_cased_pt_lenerbr
date: 2022-08-03
tags: [bert, ner, open_source, pt]
task: Named Entity Recognition
language: pt
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
annotator: BertForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `ner-bert-base-cased-pt-lenerbr` is a Portugese model originally trained by `mateusqc`.

## Predicted Entities

`PESSOA`, `LOCAL`, `TEMPO`, `LEGISLACAO`, `JURISPRUDENCIA`, `ORGANIZACAO`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_ner_ner_bert_base_cased_pt_lenerbr_pt_4.1.0_3.0_1659514894541.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_ner_ner_bert_base_cased_pt_lenerbr_pt_4.1.0_3.0_1659514894541.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

tokenClassifier = BertForTokenClassification.pretrained("bert_ner_ner_bert_base_cased_pt_lenerbr","pt") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("ner")

pipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, tokenClassifier])

data = spark.createDataFrame([["Eu amo Spark NLP"]]).toDF("text")

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

val tokenClassifier = BertForTokenClassification.pretrained("bert_ner_ner_bert_base_cased_pt_lenerbr","pt") 
    .setInputCols(Array("sentence", "token")) 
    .setOutputCol("ner")

val pipeline = new Pipeline().setStages(Array(documentAssembler,sentenceDetector, tokenizer, tokenClassifier))

val data = Seq("Eu amo Spark NLP").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("pt.ner.bert.cased_base").predict("""Eu amo Spark NLP""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_ner_ner_bert_base_cased_pt_lenerbr|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|pt|
|Size:|406.5 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/mateusqc/ner-bert-base-cased-pt-lenerbr