---
layout: model
title: Swedish BertForTokenClassification Base Cased model (from KBLab)
author: John Snow Labs
name: bert_ner_bert_base_swedish_cased_neriob
date: 2022-07-20
tags: [bert, ner, open_source, sv]
task: Named Entity Recognition
language: sv
edition: Spark NLP 4.0.0
spark_version: 3.0
supported: true
annotator: BertForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `bert-base-swedish-cased-neriob` is a Swedish model originally trained by `KBLab`.

## Predicted Entities

`PER`, `LOC`, `LOCORG`, `EVN`, `TME`, `WRK`, `MSR`, `OBJ`, `PRSWRK`, `OBJORG`, `ORG`, `ORGPRS`, `LOCPRS`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_ner_bert_base_swedish_cased_neriob_sv_4.0.0_3.0_1658319738479.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_ner_bert_base_swedish_cased_neriob_sv_4.0.0_3.0_1658319738479.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

tokenClassifier = BertForTokenClassification.pretrained("bert_ner_bert_base_swedish_cased_neriob","sv") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("ner")

pipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, tokenClassifier])

data = spark.createDataFrame([["Jag älskar Spark NLP"]]).toDF("text")

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

val tokenClassifier = BertForTokenClassification.pretrained("bert_ner_bert_base_swedish_cased_neriob","sv")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("ner")

val pipeline = new Pipeline().setStages(Array(documentAssembler,sentenceDetector, tokenizer, tokenClassifier))

val data = Seq("Jag älskar Spark NLP").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("sv.ner.bert.cased_base.neriob.by_kblab").predict("""Jag älskar Spark NLP""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_ner_bert_base_swedish_cased_neriob|
|Compatibility:|Spark NLP 4.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|sv|
|Size:|465.8 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/KBLab/bert-base-swedish-cased-neriob