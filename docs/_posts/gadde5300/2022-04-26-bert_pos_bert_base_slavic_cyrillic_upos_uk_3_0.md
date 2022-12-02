---
layout: model
title: Ukrainian Part of Speech Tagger (Base)
author: John Snow Labs
name: bert_pos_bert_base_slavic_cyrillic_upos
date: 2022-04-26
tags: [bert, pos, part_of_speech, uk, open_source]
task: Part of Speech Tagging
language: uk
edition: Spark NLP 3.4.2
spark_version: 3.0
supported: true
annotator: BertForTokenClassification
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Part of Speech model, uploaded to Hugging Face, adapted and imported into Spark NLP. `bert-base-slavic-cyrillic-upos` is a Ukrainian model orginally trained by `KoichiYasuoka`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_pos_bert_base_slavic_cyrillic_upos_uk_3.4.2_3.0_1650993899127.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

tokenClassifier = BertForTokenClassification.pretrained("bert_pos_bert_base_slavic_cyrillic_upos","uk") \
.setInputCols(["sentence", "token"]) \
.setOutputCol("pos")

pipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, tokenClassifier])

data = spark.createDataFrame([["Я люблю Spark NLP"]]).toDF("text")

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

val tokenClassifier = BertForTokenClassification.pretrained("bert_pos_bert_base_slavic_cyrillic_upos","uk") 
.setInputCols(Array("sentence", "token")) 
.setOutputCol("pos")

val pipeline = new Pipeline().setStages(Array(documentAssembler,sentenceDetector, tokenizer, tokenClassifier))

val data = Seq("Я люблю Spark NLP").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("uk.pos.bert_base_slavic_cyrillic_upos").predict("""Я люблю Spark NLP""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_pos_bert_base_slavic_cyrillic_upos|
|Compatibility:|Spark NLP 3.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|uk|
|Size:|668.2 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/KoichiYasuoka/bert-base-slavic-cyrillic-upos
- https://universaldependencies.org/be/
- https://universaldependencies.org/bg/
- https://universaldependencies.org/ru/
- https://universaldependencies.org/treebanks/sr_set/
- https://universaldependencies.org/treebanks/uk_iu/
- https://universaldependencies.org/u/pos/
- https://github.com/KoichiYasuoka/esupar