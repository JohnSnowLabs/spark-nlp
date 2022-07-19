---
layout: model
title: Detect Legal Entities in German
author: John Snow Labs
name: ner_legal
date: 2020-09-28
task: Named Entity Recognition
language: de
edition: Spark NLP for Healthcare 2.6.0
spark_version: 2.4
tags: [ner, de, licensed]
supported: true
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description
This model can be used to detect legal entities in German text.
## Predicted Entities
`STR`, `LIT`, `PER`, `EUN`, `VT`, `MRK`, `INN`, `UN`, `RS`, `ORG`, `GS`, `VS`, `LDS`, `GRT`, `VO`, `RR`, `LD`, `AN`, `ST`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_LEGAL_DE/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_LEGAL_DE.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_legal_de_2.5.5_2.4_1599471454959.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
## How to use

Use as part of an nlp pipeline with the following stages: DocumentAssembler, SentenceDetector, Tokenizer, WordEmbeddingsModel, NerDLModel. Add the NerConverter to the end of the pipeline to convert entity tokens into full entity chunks.

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPythonNLU.html %}


```python
...
word_embeddings = WordEmbeddingsModel.pretrained("w2v_cc_300d",'de','clinical/models')\
.setInputCols(["sentence", 'token'])\
.setOutputCol("embeddings")\
.setCaseSensitive(False)
legal_ner = NerDLModel.pretrained("ner_legal",'de','clinical/models') \
.setInputCols(["sentence", "token", "embeddings"]) \
.setOutputCol("ner")
...
legal_pred_pipeline = Pipeline(stages = [document_assembler, sentence_detector, tokenizer, word_embeddings, legal_ner, ner_converter])
legal_light_model = LightPipeline(legal_pred_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))

result = legal_light_model.fullAnnotate('''Jedoch wird der Verkehr darin naheliegend den Namen eines der bekanntesten Flüsse Deutschlands erkennen, welcher als Seitenfluss des Rheins durch Oberfranken, Unterfranken und Südhessen fließt und bei Mainz in den Rhein mündet. Klein , in : Maunz / Schmidt-Bleibtreu / Klein / Bethge , BVerfGG , § 19 Rn. 9 Richtlinien zur Bewertung des Grundvermögens – BewRGr – vom19. I September 1966 (BStBl I, S.890) ''')
```

```scala
...
val word_embeddings = WordEmbeddingsModel.pretrained("w2v_cc_300d","de","clinical/models")
.setInputCols(Array("sentence","token"))
.setOutputCol("embeddings")
val ner = NerDLModel.pretrained("ner_legal",'de','clinical/models')
.setInputCols("sentence", "token", "embeddings")
.setOutputCol("ner")
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, ner, ner_converter))
val data = Seq("Jedoch wird der Verkehr darin naheliegend den Namen eines der bekanntesten Flüsse Deutschlands erkennen, welcher als Seitenfluss des Rheins durch Oberfranken, Unterfranken und Südhessen fließt und bei Mainz in den Rhein mündet. Klein , in : Maunz / Schmidt-Bleibtreu / Klein / Bethge , BVerfGG , § 19 Rn. 9 Richtlinien zur Bewertung des Grundvermögens – BewRGr – vom19. I September 1966 (BStBl I, S.890)").toDF("text")
val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("de.med_ner.legal").predict("""Jedoch wird der Verkehr darin naheliegend den Namen eines der bekanntesten Flüsse Deutschlands erkennen, welcher als Seitenfluss des Rheins durch Oberfranken, Unterfranken und Südhessen fließt und bei Mainz in den Rhein mündet. Klein , in : Maunz / Schmidt-Bleibtreu / Klein / Bethge , BVerfGG , § 19 Rn. 9 Richtlinien zur Bewertung des Grundvermögens – BewRGr – vom19. I September 1966 (BStBl I, S.890) """)
```

</div>

{:.h2_title}
## Results

```bash
+---+---------------------------------------------------+----------+
| # |                                            Chunks | Entities |
+---+---------------------------------------------------+----------+
| 0 |                                      Deutschlands |       LD |
+---+---------------------------------------------------+----------+
| 1 |                                            Rheins |      LDS |
+---+---------------------------------------------------+----------+
| 2 |                                       Oberfranken |      LDS |
+---+---------------------------------------------------+----------+
| 3 |                                      Unterfranken |      LDS |
+---+---------------------------------------------------+----------+
| 4 |                                         Südhessen |      LDS |
+---+---------------------------------------------------+----------+
| 5 |                                             Mainz |       ST |
+---+---------------------------------------------------+----------+
| 6 |                                             Rhein |      LDS |
+---+---------------------------------------------------+----------+
| 7 | Klein , in : Maunz / Schmidt-Bleibtreu / Klein... |      LIT |
+---+---------------------------------------------------+----------+
| 8 | Richtlinien zur Bewertung des Grundvermögens –... |       VS |
+---+---------------------------------------------------+----------+
| 9 |                 I September 1966 (BStBl I, S.890) |       VS |
+---+---------------------------------------------------+----------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_legal|
|Type:|ner|
|Compatibility:|Spark NLP for Healthcare 2.6.0 +|
|Edition:|Official|
|License:|Licensed|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|[de]|
|Case sensitive:|false|

{:.h2_title}
## Data Source

The dataset used to train this model is taken from Leitner, et.al (2019)

Leitner, E., Rehm, G., and Moreno-Schneider, J. (2019). Fine-grained Named Entity Recognition in Legal Documents. In Maribel Acosta, et al., editors, Semantic Systems. The Power of AI and Knowledge Graphs. Proceedings of the 15th International Conference (SEMANTiCS2019), number 11702 in Lecture Notes in Computer Science, pages 272–287, Karlsruhe, Germany, 9. Springer. 10/11 September 2019.

Source of the annotated text:

Court decisions from 2017 and 2018 were selected for the dataset, published online by the Federal Ministry of Justice and Consumer Protection. The documents originate from seven federal courts: Federal Labour Court (BAG), Federal Fiscal Court (BFH), Federal Court of Justice (BGH), Federal Patent Court (BPatG), Federal Social Court (BSG), Federal Constitutional Court (BVerfG) and Federal Administrative Court (BVerwG).

## Benchmarking

```bash
+---------------+-------+------------+------+-------------+-----+------------+
| Macro-average | prec: | 0.9210195, | rec: | 0.91861916, | f1: | 0.91981775 |
+---------------+-------+------------+------+-------------+-----+------------+
| Micro-average | prec: | 0.9833763, | rec: | 0.9837547,  | f1: | 0.9835655  |
+---------------+-------+------------+------+-------------+-----+------------+
```