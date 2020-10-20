---
layout: model
title: NER DL Model Legal
author: John Snow Labs
name: ner_legal
class: NerDLModel
language: de
repository: clinical/models
date: 2020-09-07
tags: [legal,ner,de]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Named Entity recognition annotator allows for a generic model to be trained by utilizing a deep learning algorithm (Char CNNs - BiLSTM - CRF - word embeddings) inspired on a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM,CNN.

## Predicted Entities 
AN,EUN,GRT,GS,INN,LD,LDS,LIT,MRK,ORG,PER,RR,RS,ST,STR,UN,VO,VS,VT

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/15.German_Legal_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_legal_de_2.5.5_2.4_1599471454959.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
...
word_embeddings = WordEmbeddingsModel.pretrained("w2v_cc_300d",'de','clinical/models')\
  .setInputCols(["sentence", 'token'])\
  .setOutputCol("embeddings")\
  .setCaseSensitive(False)

legal_ner = NerDLModel.pretrained("ner_legal",'de','clinical/models') \
  .setInputCols(["sentence", "token", "embeddings"]) \
  .setOutputCol("ner")

legal_ner_converter = NerConverterInternal() \
  .setInputCols(["sentence", "token", "ner"]) \
  .setOutputCol("ner_chunk")\


legal_pred_pipeline = Pipeline(
      stages = [
      documentAssembler,
      sentenceDetector,
      tokenizer,
      word_embeddings,
      legal_ner,
      legal_ner_converter
      ])

legal_light_model = LightPipeline(legal_pred_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))

result = legal_light_model.fullAnnotate('''Jedoch wird der Verkehr darin naheliegend den Namen eines der bekanntesten Flüsse Deutschlands erkennen, welcher als Seitenfluss des Rheins durch Oberfranken, Unterfranken und Südhessen fließt und bei Mainz in den Rhein mündet. Klein , in : Maunz / Schmidt-Bleibtreu / Klein / Bethge , BVerfGG , § 19 Rn. 9 Richtlinien zur Bewertung des Grundvermögens – BewRGr – vom19. I September 1966 (BStBl I, S.890) ''')
```

```scala

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
|---------------|----------------------------------|
| Name:          | ner_legal                        |
| Type:   | NerDLModel                       |
| Compatibility: | Spark NLP 2.5.5+                            |
| License:       | Licensed                         |
| Edition:       | Legal                            |
|Input labels:        | [sentence, token, word_embeddings] |
|Output labels:       | [ner]                              |
| Language:      | de                               |
| Dependencies: | embeddings_clinical              |

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