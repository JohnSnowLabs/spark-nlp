---
layout: model
title: Indonesian Part of Speech Tagger (from w11wo)
author: John Snow Labs
name: roberta_pos_indonesian_roberta_base_posp_tagger
date: 2022-05-03
tags: [roberta, pos, part_of_speech, id, open_source]
task: Part of Speech Tagging
language: id
edition: Spark NLP 3.4.2
spark_version: 3.0
supported: true
annotator: RoBertaForTokenClassification
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Part of Speech model, uploaded to Hugging Face, adapted and imported into Spark NLP. `indonesian-roberta-base-posp-tagger` is a Indonesian model orginally trained by `w11wo`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_pos_indonesian_roberta_base_posp_tagger_id_3.4.2_3.0_1651596272433.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

tokenClassifier = RoBertaForTokenClassification.pretrained("roberta_pos_indonesian_roberta_base_posp_tagger","id") \
.setInputCols(["sentence", "token"]) \
.setOutputCol("pos")

pipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, tokenClassifier])

data = spark.createDataFrame([["Saya suka Spark NLP"]]).toDF("text")

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

val tokenClassifier = RoBertaForTokenClassification.pretrained("roberta_pos_indonesian_roberta_base_posp_tagger","id") 
.setInputCols(Array("sentence", "token")) 
.setOutputCol("pos")

val pipeline = new Pipeline().setStages(Array(documentAssembler,sentenceDetector, tokenizer, tokenClassifier))

val data = Seq("Saya suka Spark NLP").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("id.pos.indonesian_roberta_base_posp_tagger").predict("""Saya suka Spark NLP""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_pos_indonesian_roberta_base_posp_tagger|
|Compatibility:|Spark NLP 3.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|id|
|Size:|466.1 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/w11wo/indonesian-roberta-base-posp-tagger
- https://arxiv.org/abs/1907.11692
- https://hf.co/flax-community/indonesian-roberta-base
- https://hf.co/datasets/indonlu
- https://w11wo.github.io/