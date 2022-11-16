---
layout: model
title: French CamembertForTokenClassification Cased model (from taln-ls2n)
author: John Snow Labs
name: camembert_classifier_poet
date: 2022-09-25
tags: [camembert, pos, open_source, fr]
task: Part of Speech Tagging
language: fr
edition: Spark NLP 4.2.0
spark_version: 3.0
supported: true
annotator: CamemBertForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained CamembertForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `POET` is a French model originally trained by `taln-ls2n`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/camembert_classifier_poet_fr_4.2.0_3.0_1664084377226.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

sequenceClassifier_loaded = CamemBertForTokenClassification.pretrained("camembert_classifier_poet","fr") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("pos")

pipeline = Pipeline(stages=[documentAssembler,sentenceDetector,tokenizer,sequenceClassifier_loaded])

data = spark.createDataFrame([["J'adore Spark NLP"]]).toDF("text")

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

val sequenceClassifier_loaded = CamemBertForTokenClassification.pretrained("camembert_classifier_poet","fr") 
    .setInputCols(Array("sentence", "token")) 
    .setOutputCol("pos")

val pipeline = new Pipeline().setStages(Array(documentAssembler,sentenceDetector,tokenizer,sequenceClassifier_loaded))

val data = Seq("J'adore Spark NLP").toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|camembert_classifier_poet|
|Compatibility:|Spark NLP 4.2.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|fr|
|Size:|410.2 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

- https://huggingface.co/taln-ls2n/POET
- https://github.com/qanastek/ANTILLES
- https://arxiv.org/abs/1911.03894
- https://www.linkedin.com/in/yanis-labrak-8a7412145/
- https://cv.archives-ouvertes.fr/richard-dufour
- https://lia.univ-avignon.fr/
- https://www.ls2n.fr/equipe/taln/
- https://pypi.org/project/transformers/
- https://universaldependencies.org/treebanks/fr_gsd/index.html
- https://github.com/ryanmcd/uni-dep-tb
- http://pageperso.lif.univ-mrs.fr/frederic.bechet/download.html
- http://pageperso.lif.univ-mrs.fr/frederic.bechet/index-english.html
- https://github.com/qanastek/ANTILLES
- https://universaldependencies.org/format.html
- https://github.com/qanastek/ANTILLES/blob/main/ANTILLES/test.conllu
- https://zenidoc.fr/
- https://anr-diets.univ-avignon.fr
- https://anr.fr/en/funded-projects-and-impact/funded-projects/project/funded/project/b2d9d3668f92a3b9fbbf7866072501ef-fd7e69d902/?tx_anrprojects_funded%5Bcontroller%5D=Funded&cHash=cb6d54d24c9e21e0d50fabf46bd56646