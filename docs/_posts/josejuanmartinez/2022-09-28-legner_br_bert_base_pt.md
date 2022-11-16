---
layout: model
title: Brazilian Portuguese NER for Laws (Bert, Base)
author: John Snow Labs
name: legner_br_bert_base
date: 2022-09-28
tags: [pt, legal, ner, laws, licensed]
task: Named Entity Recognition
language: pt
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
annotator: BertForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is a Deep Learning Portuguese Named Entity Recognition model for the legal domain, trained using Base Bert Embeddings, and is able to predict the following entities:

- ORGANIZACAO (Organizations)
- JURISPRUDENCIA (Jurisprudence)
- PESSOA (Person)
- TEMPO (Time)
- LOCAL (Location)
- LEGISLACAO (Laws)
- O (Other)

You can find different versions of this model in Models Hub:
- With a Deep Learning architecture (non-transformer) and Base Embeddings;
- With a Deep Learning architecture (non-transformer) and Large Embeddings;
- With a Transformers Architecture and Base Embeddings;
- With a Transformers Architecture and Large Embeddings;

## Predicted Entities

`PESSOA`, `ORGANIZACAO`, `LEGISLACAO`, `JURISPRUDENCIA`, `TEMPO`, `LOCAL`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_LEGAL_PT/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legner_br_bert_base_pt_1.0.0_3.0_1664362186486.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = nlp.DocumentAssembler() \
        .setInputCol("text") \
        .setOutputCol("document")

sentenceDetector = nlp.SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")\
       .setInputCols(["document"])\
       .setOutputCol("sentence")

tokenizer = nlp.Tokenizer() \
    .setInputCols("sentence") \
    .setOutputCol("token")

tokenClassifier = nlp.BertForTokenClassification.pretrained("legner_br_bert_base","pt", "legal/models") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("ner")

pipeline = Pipeline(
  stages=[
    documentAssembler, 
    sentenceDetector, 
    tokenizer, 
    tokenClassifier])

example = spark.createDataFrame(pd.DataFrame({'text': ["""Mediante do exposto , com fundamento nos artigos 32 , i , e 33 , da lei 8.443/1992 , submetem-se os autos à consideração superior , com posterior encaminhamento ao ministério público junto ao tcu e ao gabinete do relator , propondo : a ) conhecer do recurso e , no mérito , negar-lhe provimento ; b ) comunicar ao recorrente , ao superior tribunal militar e ao tribunal regional federal da 2ª região , a fim de fornecer subsídios para os processos judiciais 2001.34.00.024796-9 e 2003.34.00.044227-3 ; e aos demais interessados a deliberação que vier a ser proferida por esta corte ” ."""]}))

result = pipeline.fit(example).transform(example)
```

</div>

## Results

```bash
+-------------------+----------------+
|              token|             ner|
+-------------------+----------------+
|             diante|               O|
|                 do|               O|
|            exposto|               O|
|                  ,|               O|
|                com|               O|
|         fundamento|               O|
|                nos|               O|
|            artigos|    B-LEGISLACAO|
|                 32|    I-LEGISLACAO|
|                  ,|    I-LEGISLACAO|
|                  i|    I-LEGISLACAO|
|                  ,|    I-LEGISLACAO|
|                  e|    I-LEGISLACAO|
|                 33|    I-LEGISLACAO|
|                  ,|    I-LEGISLACAO|
|                 da|    I-LEGISLACAO|
|                lei|    I-LEGISLACAO|
|         8.443/1992|    I-LEGISLACAO|
|                  ,|               O|
|        submetem-se|               O|
|                 os|               O|
|              autos|               O|
|                  à|               O|
|       consideração|               O|
|           superior|               O|
|                  ,|               O|
|                com|               O|
|          posterior|               O|
|     encaminhamento|               O|
|                 ao|               O|
|         ministério|   B-ORGANIZACAO|
|            público|   I-ORGANIZACAO|
|              junto|               O|
|                 ao|               O|
|                tcu|   B-ORGANIZACAO|
|                  e|               O|
|                 ao|               O|
|           gabinete|               O|
|                 do|               O|
|            relator|               O|
|                  ,|               O|
|           propondo|               O|
|                  :|               O|
|                  a|               O|
|                  )|               O|
|           conhecer|               O|
|                 do|               O|
|            recurso|               O|
|                  e|               O|
|                  ,|               O|
|                 no|               O|
|             mérito|               O|
|                  ,|               O|
|          negar-lhe|               O|
|         provimento|               O|
|                  ;|               O|
|                  b|               O|
|                  )|               O|
|          comunicar|               O|
|                 ao|               O|
|         recorrente|               O|
|                  ,|               O|
|                 ao|               O|
|           superior|   B-ORGANIZACAO|
|           tribunal|   I-ORGANIZACAO|
|            militar|   I-ORGANIZACAO|
|                  e|               O|
|                 ao|               O|
|           tribunal|   B-ORGANIZACAO|
|           regional|   I-ORGANIZACAO|
|            federal|   I-ORGANIZACAO|
|                 da|   I-ORGANIZACAO|
|                 2ª|   I-ORGANIZACAO|
|             região|   I-ORGANIZACAO|
|                  ,|               O|
|                  a|               O|
|                fim|               O|
|                 de|               O|
|           fornecer|               O|
|          subsídios|               O|
|               para|               O|
|                 os|               O|
|          processos|               O|
|          judiciais|               O|
|2001.34.00.024796-9|B-JURISPRUDENCIA|
|                  e|               O|
|2003.34.00.044227-3|B-JURISPRUDENCIA|
|                  ;|               O|
|                  e|               O|
|                aos|               O|
|             demais|               O|
|       interessados|               O|
|                  a|               O|
|        deliberação|               O|
|                que|               O|
|               vier|               O|
|                  a|               O|
|                ser|               O|
|          proferida|               O|
|                por|               O|
|               esta|               O|
|              corte|               O|
|                  ”|               O|
|                  .|               O|
+-------------------+----------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legner_br_bert_base|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|pt|
|Size:|406.5 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

Original texts available in https://paperswithcode.com/sota?task=Token+Classification&dataset=lener_br and in-house data augmentation with weak labelling

## Benchmarking

```bash
           label  precision    recall  f1-score   support
   B-ORGANIZACAO       0.86      0.86      0.86       499
   I-ORGANIZACAO       0.89      0.89      0.89       859
    B-LEGISLACAO       0.94      0.94      0.94       373
    I-LEGISLACAO       0.96      0.98      0.97      2235
B-JURISPRUDENCIA       0.76      0.54      0.63       183
I-JURISPRUDENCIA       0.87      0.79      0.83       475
         B-TEMPO       0.92      0.61      0.74       192
         I-TEMPO       0.90      0.93      0.91        68
        B-PESSOA       0.93      0.96      0.95       231
        I-PESSOA       0.96      0.99      0.97       494
         B-LOCAL       0.78      0.81      0.79        47
         I-LOCAL       0.59      0.74      0.66        85
       micro-avg       0.91      0.91      0.91      5741
       macro-avg       0.86      0.84      0.84      5741
    weighted-avg       0.91      0.91      0.91      5741
```