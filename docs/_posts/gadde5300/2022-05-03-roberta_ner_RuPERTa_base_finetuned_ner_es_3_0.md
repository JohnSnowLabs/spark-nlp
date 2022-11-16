---
layout: model
title: Spanish Named Entity Recognition (from mrm8488)
author: John Snow Labs
name: roberta_ner_RuPERTa_base_finetuned_ner
date: 2022-05-03
tags: [roberta, ner, open_source, es]
task: Named Entity Recognition
language: es
edition: Spark NLP 3.4.2
spark_version: 3.0
supported: true
annotator: RoBertaForTokenClassification
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Named Entity Recognition model, uploaded to Hugging Face, adapted and imported into Spark NLP. `RuPERTa-base-finetuned-ner` is a Spanish model orginally trained by `mrm8488`.

## Predicted Entities

`MISC`, `ORG`, `LOC`, `PER`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_ner_RuPERTa_base_finetuned_ner_es_3.4.2_3.0_1651593655495.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

tokenClassifier = RoBertaForTokenClassification.pretrained("roberta_ner_RuPERTa_base_finetuned_ner","es") \
.setInputCols(["sentence", "token"]) \
.setOutputCol("ner")

pipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, tokenClassifier])

data = spark.createDataFrame([["Amo Spark NLP"]]).toDF("text")

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

val tokenClassifier = RoBertaForTokenClassification.pretrained("roberta_ner_RuPERTa_base_finetuned_ner","es") 
.setInputCols(Array("sentence", "token")) 
.setOutputCol("ner")

val pipeline = new Pipeline().setStages(Array(documentAssembler,sentenceDetector, tokenizer, tokenClassifier))

val data = Seq("Amo Spark NLP").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("es.ner.RuPERTa_base_finetuned_ner").predict("""Amo Spark NLP""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_ner_RuPERTa_base_finetuned_ner|
|Compatibility:|Spark NLP 3.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|es|
|Size:|470.7 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/mrm8488/RuPERTa-base-finetuned-ner
- https://www.kaggle.com/nltkdata/conll-corpora
- https://www.kaggle.com/nltkdata/conll-corpora
- https://twitter.com/mrm8488