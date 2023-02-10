---
layout: model
title: German NER for Laws
author: John Snow Labs
name: legner_courts
date: 2022-10-02
tags: [de, legal, ner, laws, court, licensed]
task: Named Entity Recognition
language: de
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
annotator: LegalNerModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model can be used to detect legal entities in German text, predicting up to 19 different labels:
```
| tag	| meaning 
-----------------
| AN	| Anwalt 
| EUN	| Europäische Norm 
| GS	| Gesetz 
| GRT	| Gericht 
| INN	| Institution 
| LD	| Land 
| LDS	| Landschaft 
| LIT	| Literatur 
| MRK	| Marke 
| ORG	| Organisation 
| PER	| Person 
| RR	| Richter 
| RS	| Rechtssprechung 
| ST	| Stadt 
| STR	| Straße 
| UN	| Unternehmen 
| VO	| Verordnung 
| VS	| Vorschrift 
| VT	| Vertrag 
```

German Named Entity Recognition model, trained using a Deep Learning architecture (CharCNN + LSTM) with a Court Decisions (2017-2018) dataset (check `Data Source` section). You can also find transformer-based versions of this model in our Models Hub (`legner_bert_base_courts` and `legner_bert_large_courts`)

## Predicted Entities

`STR`, `LIT`, `PER`, `EUN`, `VT`, `MRK`, `INN`, `UN`, `RS`, `ORG`, `GS`, `VS`, `LDS`, `GRT`, `VO`, `RR`, `LD`, `AN`, `ST`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_LEGAL_DE/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legner_courts_de_1.0.0_3.0_1664706079878.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legner_courts_de_1.0.0_3.0_1664706079878.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
word_embeddings = nlp.WordEmbeddingsModel.pretrained("w2v_cc_300d",'de','clinical/models')\
.setInputCols(["sentence", 'token'])\
.setOutputCol("embeddings")\
.setCaseSensitive(False)

legal_ner = legal.NerModel.pretrained("legner_courts",'de','legal/models') \
.setInputCols(["sentence", "token", "embeddings"]) \
.setOutputCol("ner")
...

legal_pred_pipeline = Pipeline(stages = [document_assembler, sentence_detector, tokenizer, word_embeddings, legal_ner, ner_converter])
legal_light_model = LightPipeline(legal_pred_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))

result = legal_light_model.fullAnnotate('''Jedoch wird der Verkehr darin naheliegend den Namen eines der bekanntesten Flüsse Deutschlands erkennen, welcher als Seitenfluss des Rheins durch Oberfranken, Unterfranken und Südhessen fließt und bei Mainz in den Rhein mündet. Klein , in : Maunz / Schmidt-Bleibtreu / Klein / Bethge , BVerfGG , § 19 Rn. 9 Richtlinien zur Bewertung des Grundvermögens – BewRGr – vom19. I September 1966 (BStBl I, S.890) ''')
```

</div>

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
|Model Name:|legner_courts|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|de|
|Size:|15.0 MB|

## References

The dataset used to train this model is taken from Leitner, et.al (2019)

Leitner, E., Rehm, G., and Moreno-Schneider, J. (2019). Fine-grained Named Entity Recognition in Legal Documents. In Maribel Acosta, et al., editors, Semantic Systems. The Power of AI and Knowledge Graphs. Proceedings of the 15th International Conference (SEMANTiCS2019), number 11702 in Lecture Notes in Computer Science, pages 272–287, Karlsruhe, Germany, 9. Springer. 10/11 September 2019.

Source of the annotated text:

Court decisions from 2017 and 2018 were selected for the dataset, published online by the Federal Ministry of Justice and Consumer Protection. The documents originate from seven federal courts: Federal Labour Court (BAG), Federal Fiscal Court (BFH), Federal Court of Justice (BGH), Federal Patent Court (BPatG), Federal Social Court (BSG), Federal Constitutional Court (BVerfG) and Federal Administrative Court (BVerwG).

## Benchmarking

```bash
        label        prec         rec          f1
Macro-average   0.9210195   0.9186192   0.9198177
Micro-average   0.9833763   0.9837547   0.9835655 
```