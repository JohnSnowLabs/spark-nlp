---
layout: model
title: Spanish Part of Speech Tagger (from mrm8488)
author: John Snow Labs
name: roberta_pos_RuPERTa_base_finetuned_pos
date: 2022-05-03
tags: [roberta, pos, part_of_speech, es, open_source]
task: Part of Speech Tagging
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

Pretrained Part of Speech model, uploaded to Hugging Face, adapted and imported into Spark NLP. `RuPERTa-base-finetuned-pos` is a Spanish model orginally trained by `mrm8488`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_pos_RuPERTa_base_finetuned_pos_es_3.4.2_3.0_1651596033465.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

tokenClassifier = RoBertaForTokenClassification.pretrained("roberta_pos_RuPERTa_base_finetuned_pos","es") \
.setInputCols(["sentence", "token"]) \
.setOutputCol("pos")

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

val tokenClassifier = RoBertaForTokenClassification.pretrained("roberta_pos_RuPERTa_base_finetuned_pos","es") 
.setInputCols(Array("sentence", "token")) 
.setOutputCol("pos")

val pipeline = new Pipeline().setStages(Array(documentAssembler,sentenceDetector, tokenizer, tokenClassifier))

val data = Seq("Amo Spark NLP").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("es.pos.RuPERTa_base_finetuned_pos").predict("""Amo Spark NLP""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_pos_RuPERTa_base_finetuned_pos|
|Compatibility:|Spark NLP 3.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|es|
|Size:|470.8 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/mrm8488/RuPERTa-base-finetuned-pos
- https://www.kaggle.com/nltkdata/conll-corpora
- https://www.kaggle.com/nltkdata/conll-corpora
- https://twitter.com/mrm8488
- https://www.linkedin.com/in/manuel-romero-cs/