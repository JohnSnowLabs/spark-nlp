---
layout: model
title: Legal Multilabel Classification (MultiEURLEX, German)
author: John Snow Labs
name: legmulticlf_multieurlex_german
date: 2023-03-24
tags: [legal, classification, de, licensed, multieurlex, open_source, tensorflow]
task: Text Classification
language: de
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
engine: tensorflow
annotator: MultiClassifierDLModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a Multilabel Text Classification model that can help you classify 17 types of  French legal documents.

## Predicted Entities

`Aarhus (Amt)`, `parlamentarischer Ausschuss`, `Veruntreuung`, `Großhandel`, `Binnenhandel`, `Handel`, `Waffenhandel`, `Klebstoff`, `öffentliche Auftragsvergabe`, `Außenhandel`, `Staatshandel`, `Einzelhandel`, `Kommission UNO`, `Untersuchungsausschuss`, `internationaler Handel`, `Ausschuss EP`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/legmulticlf_multieurlex_german_de_1.0.0_3.0_1679669521877.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/legmulticlf_multieurlex_german_de_1.0.0_3.0_1679669521877.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = nlp.DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")\
    .setCleanupMode("shrink")

embeddings = nlp.UniversalSentenceEncoder.pretrained()\
    .setInputCols("document")\
    .setOutputCol("sentence_embeddings")

docClassifier = nlp.MultiClassifierDLModel().pretrained("legmulticlf_multieurlex_german", "de", "legal/models")\
    .setInputCols("sentence_embeddings") \
    .setOutputCol("class")

pipeline = nlp.Pipeline(
    stages=[
        document_assembler,
        embeddings,
        docClassifier
    ]
)

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = pipeline.fit(empty_data)

light_model = nlp.LightPipeline(model)

result = light_model.annotate("""ENTSCHEIDUNG DER KOMMISSION
vom 6. März 2006
zur Festlegung der Brandverhaltensklassen für bestimmte Bauprodukte (Holzfußböden sowie Wand- und Deckenbekleidungen aus Massivholz)
(Bekannt gegeben unter Aktenzeichen K(2006) 655)
(Text von Bedeutung für den EWR)
(2006/213/EG)
DIE KOMMISSION DER EUROPÄISCHEN GEMEINSCHAFTEN -
gestützt auf den Vertrag zur Gründung der Europäischen Gemeinschaft,
gestützt auf die Richtlinie 89/106/EWG des Rates vom 21. Dezember 1988 zur Angleichung der Rechts- und Verwaltungsvorschriften der Mitgliedstaaten über Bauprodukte (1), insbesondere auf Artikel 20 Absatz 2,
in Erwägung nachstehender Gründe:
(1)
Nach der Richtlinie 89/106/EWG kann es zur Berücksichtigung der auf einzelstaatlicher, regionaler oder lokaler Ebene bestehenden unterschiedlichen Schutzniveaus für Bauwerke erforderlich sein, dass in den Grundlagendokumenten Klassen entsprechend der Leistung des jeweiligen Produkts im Hinblick auf die jeweilige wesentliche Anforderung festgelegt werden. Diese Dokumente wurden in Form einer Mitteilung der Kommission über die Grundlagendokumente der Richtlinie 89/106/EWG des Rates (2) veröffentlicht.
(2)
Für die wesentliche Anforderung „Brandschutz“ enthält das Grundlagendokument Nr. 2 eine Reihe zusammenhängender Maßnahmen, die gemeinsam die Strategie für den Brandschutz festlegen, die dann in den Mitgliedstaaten in unterschiedlicher Weise entwickelt werden kann.
(3)
Das Grundlagendokument Nr. 2 nennt als eine dieser Maßnahmen die Begrenzung der Entstehung und Ausbreitung von Feuer und Rauch in einem gegebenen Bereich, indem das Potenzial der Bauprodukte, zu einem Vollbrand beizutragen, begrenzt wird.
(4)
Das Grenzniveau kann nur in Form unterschiedlicher Stufen des Brandverhaltens der Bauprodukte in ihrer Endanwendung ausgedrückt werden.
(5)
Als harmonisierte Lösung wurde in der Entscheidung 2000/147/EG der Kommission vom 8. Februar 2000 zur Durchführung der Richtlinie 89/106/EWG des Rates im Hinblick auf die Klassifizierung des Brandverhaltens von Bauprodukten (3) ein System von Klassen festgelegt.
(6)
Für Holzfußböden sowie Wand- und Deckenbekleidungen aus Massivholz muss die mit der Entscheidung 2000/147/EG festgelegte Klassifizierung verwendet werden.
(7)
Das Brandverhalten zahlreicher Bauprodukte/-materialien im Rahmen der in der Entscheidung 2000/147/EG festgelegten Klassifizierung ist so eindeutig ermittelt und den für die Brandschutzvorschriften zuständigen Stellen in den Mitgliedstaaten so gut bekannt, dass sich eine Prüfung dieses Leistungsmerkmals erübrigt.
(8)
Die in dieser Entscheidung vorgesehenen Maßnahmen entsprechen der Stellungnahme des Ständigen Ausschusses für das Bauwesen -
HAT FOLGENDE ENTSCHEIDUNG ERLASSEN:
Artikel 1
Die Bauprodukte und/oder -materialien, die alle Anforderungen des Merkmals „Brandverhalten“ erfüllen, ohne dass eine weitere Prüfung erforderlich ist, sind im Anhang aufgeführt.
Artikel 2
Die spezifischen Klassen, die im Rahmen der in der Entscheidung 2000/147/EG festgelegten Klassifizierung des Brandverhaltens für unterschiedliche Bauprodukte und/oder -materialien gelten, sind im Anhang aufgeführt.
Artikel 3
Die Produkte werden - sofern relevant - in Bezug auf ihre Endanwendung betrachtet.
Artikel 4
Diese Entscheidung ist an die Mitgliedstaaten gerichtet.
Brüssel, den 6. März 2006""")

```

</div>

## Results

```bash
Veruntreuung,Großhandel,Kommission UNO
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legmulticlf_multieurlex_german|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|de|
|Size:|12.9 MB|

## References

https://huggingface.co/datasets/nlpaueb/multi_eurlex

## Benchmarking

```bash
 
labels   precision    recall  f1-score   support

0       0.65      0.35      0.46       124
1       0.00      0.00      0.00        27
2       0.89      0.57      0.70        14
3       1.00      0.22      0.36         9
4       0.74      0.42      0.54       143
5       0.80      0.55      0.65       716
6       0.84      0.97      0.90      1092
7       0.84      0.91      0.87      1012
8       0.87      0.75      0.81        64
9       0.89      0.71      0.79        34
10       0.89      0.57      0.69       137
11       0.92      0.78      0.84       273
12       0.00      0.00      0.00         5
13       0.86      0.71      0.78        45
14       0.81      0.75      0.78       653
15       0.65      0.94      0.77        18
   micro-avg       0.83      0.78      0.80      4366
   macro-avg       0.73      0.58      0.62      4366
weighted-avg       0.82      0.78      0.79      4366
 samples-avg       0.83      0.78      0.78      4366
F1-micro-averaging: 0.8028851838713491
ROC:  0.8722362755950475

```
