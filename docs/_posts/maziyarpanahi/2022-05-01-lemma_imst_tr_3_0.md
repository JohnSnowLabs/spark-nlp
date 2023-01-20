---
layout: model
title: Lemma UD model for Turkish (lemma_imst)
author: John Snow Labs
name: lemma_imst
date: 2022-05-01
tags: [open_source, universal_dependency, lemmatizer, tr, turkish]
task: Lemmatization
language: tr
edition: Spark NLP 3.4.3
spark_version: 3.0
supported: true
annotator: LemmatizerModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Lemmatizer model (`lemma_imst`) trained on Universal Dependencies 2.9 (UD_Turkish-IMST) in Turkish language.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/annotation/english/model-downloader/Create%20custom%20pipeline%20-%20NerDL.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/lemma_imst_tr_3.4.3_3.0_1651416684397.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/lemma_imst_tr_3.4.3_3.0_1651416684397.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

lemma = LemmatizerModel.pretrained("lemma_imst", "tr")\ 
.setInputCols(["token"])\
.setOutputCol("lemma")

pipeline = Pipeline(stages=[document, sentence, tokenizer, lemma])

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

val lemma = LemmatizerModel.pretrained("lemma_imst", "tr")
.setInputCols("token")
.setOutputCol("lemma")

val pipeline = new Pipeline().setStages(Array(document, sentence, tokenizer, lemma))

val data = Seq("I love Spark NLP").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("tr.lemma.imst").predict("""I love Spark NLP""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|lemma_imst|
|Compatibility:|Spark NLP 3.4.3+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[form]|
|Output Labels:|[lemma]|
|Language:|tr|
|Size:|214.4 KB|

## References

Model is trained on Universal Dependencies (treebank 2.9) `UD_Turkish-IMST`
[https://github.com/UniversalDependencies/UD_Turkish-IMST](https://github.com/UniversalDependencies/UD_Turkish-IMST)