---
layout: model
title: Lemma UD model for Coptic (pos_scriptorium)
author: John Snow Labs
name: pos_scriptorium
date: 2022-05-01
tags: [open_source, universal_dependency, pos, part_of_speech, cop, coptic]
task: Part of Speech Tagging
language: cop
edition: Spark NLP 3.4.3
spark_version: 3.0
supported: true
annotator: PerceptronModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Lemmatizer model (`pos_scriptorium`) trained on Universal Dependencies 2.9 (UD_Coptic-Scriptorium) in Coptic language.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/GRAMMAR_EN.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_scriptorium_cop_3.4.3_3.0_1651419869890.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

document = DocumentAssembler()\ 
.setInputCol("text")\ 
.setOutputCol("document")

sentence = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")\ 
.setInputCols(["document"])\ 
.setOutputCol("sentence")

tokenizer = Tokenizer()\ 
.setInputCols(["sentence"])\ 
.setOutputCol("token") 

pos = PerceptronModel.pretrained("pos_scriptorium", "cop")\ 
.setInputCols(["sentence", "token"])\ 
.setOutputCol("pos")

pipeline = Pipeline(stages=[document, sentence, tokenizer, pos])

data = spark.createDataFrame([["I love Spark NLP"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala

val document = new DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val sentence = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")
.setInputCols("document")
.setOutputCol("sentence")

val tokenizer = new Tokenizer() 
.setInputCols("sentence") 
.setOutputCol("token")

val pos = PerceptronModel.pretrained("pos_scriptorium", "cop")
.setInputCols("sentence", "token")
.setOutputCol("pos")

val pipeline = new Pipeline().setStages(Array(document, sentence, tokenizer, pos))

val data = Seq("I love Spark NLP").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("cop.pos").predict("""I love Spark NLP""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pos_scriptorium|
|Compatibility:|Spark NLP 3.4.3+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, form]|
|Output Labels:|[pos]|
|Language:|cop|
|Size:|615.0 KB|

## References

Model is trained on Universal Dependencies (treebank 2.9) `UD_Coptic-Scriptorium`
[https://github.com/UniversalDependencies/UD_Coptic-Scriptorium](https://github.com/UniversalDependencies/UD_Coptic-Scriptorium)