---
layout: model
title: English BertForTokenClassification Cased model (from kunalr63)
author: John Snow Labs
name: bert_ner_simple_transformer
date: 2023-11-06
tags: [bert, ner, open_source, en, onnx]
task: Named Entity Recognition
language: en
edition: Spark NLP 5.2.0
spark_version: 3.0
supported: true
engine: onnx
annotator: BertForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `simple_transformer` is a English model originally trained by `kunalr63`.

## Predicted Entities

`L-CLG`, `U-LOC`, `L-SKILLS`, `U-DESIG`, `U-SKILLS`, `L-ADDRESS`, `WORK_EXP`, `U-COMPANY`, `U-PER`, `L-EMAIL`, `DESIG`, `L-PER`, `L-LOC`, `LOC`, `COMPANY`, `L-QUALI`, `L-TRAIN`, `L-COMPANY`, `SCH`, `SKILLS`, `L-DESIG`, `L-WORK_EXP`, `L-SCH`, `U-SCH`, `CLG`, `L-HOBBI`, `L-EXPERIENCE`, `TRAIN`, `CERTIFICATION`, `QUALI`, `PHONE`, `U-CLG`, `U-EXPERIENCE`, `EMAIL`, `U-PHONE`, `PER`, `U-QUALI`, `L-CERTIFICATION`, `L-PHONE`, `HOBBI`, `U-EMAIL`, `ADDRESS`, `EXPERIENCE`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_ner_simple_transformer_en_5.2.0_3.0_1699300440938.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_ner_simple_transformer_en_5.2.0_3.0_1699300440938.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

tokenClassifier = BertForTokenClassification.pretrained("bert_ner_simple_transformer","en") \
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

val tokenClassifier = BertForTokenClassification.pretrained("bert_ner_simple_transformer","en") 
    .setInputCols(Array("sentence", "token")) 
    .setOutputCol("ner")

val pipeline = new Pipeline().setStages(Array(documentAssembler,sentenceDetector, tokenizer, tokenClassifier))

val data = Seq("PUT YOUR STRING HERE").toDF("text")

val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu
nlu.load("en.ner.bert.by_kunalr63").predict("""PUT YOUR STRING HERE""")
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_ner_simple_transformer|
|Compatibility:|Spark NLP 5.2.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|407.3 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

References

- https://huggingface.co/kunalr63/simple_transformer