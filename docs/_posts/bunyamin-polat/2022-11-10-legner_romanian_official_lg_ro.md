---
layout: model
title: Named Entity Recognition in Romanian Official Documents (Large)
author: John Snow Labs
name: legner_romanian_official_lg
date: 2022-11-10
tags: [ro, ner, legal, licensed]
task: Named Entity Recognition
language: ro
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
annotator: LegalNerModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a large version NER model that extracts following 14 entities from Romanian Official Documents.

## Predicted Entities

`PER`, `LOC`, `ORG`, `DATE`, `DECISION`, `DECREE`, `DIRECTIVE`, `ORDINANCE`, `EMERGENCY_ORDINANCE`, `LAW`, `ORDER`, `REGULATION`, `REPORT`, `TREATY`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/legal/LEGNER_ROMANIAN_OFFICIAL/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legner_romanian_official_lg_ro_1.0.0_3.0_1668084251147.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = nlp.DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")\

sentence_detector = nlp.SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")\
    .setInputCols(["document"])\
    .setOutputCol("sentence")\

tokenizer = nlp.Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

embeddings = nlp.BertEmbeddings.pretrained("bert_base_cased", "ro")\
    .setInputCols("sentence", "token")\
    .setOutputCol("embeddings")\
    .setMaxSentenceLength(512)\
    .setCaseSensitive(True)

ner_model = legal.NerModel.pretrained("legner_romanian_official_lg", "ro", "legal/models")\
    .setInputCols(["sentence", "token", "embeddings"])\
    .setOutputCol("ner")\

ner_converter = nlp.NerConverter()\
    .setInputCols(["sentence", "token", "ner"])\
    .setOutputCol("ner_chunk")

pipeline = Pipeline(stages=[
    document_assembler,
    sentence_detector,
    tokenizer,
    embeddings,
    ner_model,
    ner_converter   
    ])

model = pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

data = spark.createDataFrame([["""LEGE nr. 159 din 25 iulie 2019 pentru modificarea și completarea Decretului-lege nr. 118 / 1990 privind acordarea unor drepturi persoanelor persecutate din motive politice de dictatura instaurată cu începere de la 6 martie 1945, precum și celor deportate în străinătate ori constituite în prizonieri și pentru modificarea Ordonanței Guvernului nr. 105 / 1999 privind acordarea unor drepturi persoanelor persecutate de către regimurile instaurate în România cu începere de la 6 septembrie 1940 până la 6 martie 1945 din motive etnice

Publicat în MONITORUL OFICIAL nr. 625 din 26 iulie 2019

Parlamentul României adoptă prezenta lege.
Articolul I
Decretul-lege nr. 118 / 1990 privind acordarea unor drepturi persoanelor persecutate din motive politice de dictatura instaurată cu începere de la 6 martie 1945, precum și celor deportate în străinătate ori constituite în prizonieri, republicat în Monitorul Oficial al României, Partea I, nr. 631 din 23 septembrie 2009, cu modificările și completările ulterioare, se modifică și se completează după cum urmează:

Articolul II
Ordonanța Guvernului nr. 105 / 1999 privind acordarea unor drepturi persoanelor persecutate de către regimurile instaurate în România cu începere de la 6 septembrie 1940 până la 6 martie 1945 din motive etnice, publicată în Monitorul Oficial al României, Partea I, nr. 426 din 31 august 1999, aprobată cu modificări și completări prin Legea nr. 189 / 2000, cu modificările și completările ulterioare, se modifică după cum urmează:
Această lege a fost adoptată de Parlamentul României, cu respectarea prevederilor art. 75 și ale art. 76 alin. (2) din Constituția României, republicată.
PREȘEDINTELE CAMEREI DEPUTAȚILOR
ION-MARCEL CIOLACU
București, 25 iulie 2019."""]]).toDF("text")
                             
result = model.transform(data)
```

</div>

## Results

```bash
+------------------------------------+---------+
|chunk                               |label    |
+------------------------------------+---------+
|LEGE nr. 159 din 25 iulie 2019      |LAW      |
|Decretului-lege nr. 118 / 1990      |DECREE   |
|6 martie 1945                       |DATE     |
|Ordonanței Guvernului nr. 105 / 1999|ORDINANCE|
|România                             |LOC      |
|6 septembrie 1940                   |DATE     |
|6 martie 1945                       |DATE     |
|26 iulie 2019                       |DATE     |
|Parlamentul României                |ORG      |
|Decretul-lege nr. 118 / 1990        |DECREE   |
|6 martie 1945                       |DATE     |
|Monitorul Oficial al României       |ORG      |
|23 septembrie 2009                  |DATE     |
|Ordonanța Guvernului nr. 105 / 1999 |ORDINANCE|
|România                             |LOC      |
|6 septembrie 1940                   |DATE     |
|6 martie 1945                       |DATE     |
|Monitorul Oficial al României       |ORG      |
|31 august 1999                      |DATE     |
|Legea nr. 189 / 2000                |LAW      |
|Parlamentul României                |ORG      |
|Constituția României                |LAW      |
|CAMEREI DEPUTAȚILOR                 |ORG      |
|ION-MARCEL CIOLACU                  |PER      |
|București                           |LOC      |
|25 iulie 2019                       |DATE     |
+------------------------------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legner_romanian_official_lg|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|ro|
|Size:|16.5 MB|

## References

Dataset is available [here](https://zenodo.org/record/7025333#.Y2zsquxBx83).

## Benchmarking

```bash
label                  precision  recall  f1-score  support 
DATE                   0.9036     0.9223  0.9128    193     
DECISION               0.9831     0.9831  0.9831    59      
DECREE                 0.5000     1.0000  0.6667    1       
DIRECTIVE              1.0000     0.6667  0.8000    3       
EMERGENCY_ORDINANCE    1.0000     0.9615  0.9804    26      
LAW                    0.9619     0.9806  0.9712    103     
LOC                    0.9110     0.8365  0.8721    159     
ORDER                  0.9767     1.0000  0.9882    42      
ORDINANCE              1.0000     0.9500  0.9744    20      
ORG                    0.8899     0.8879  0.8889    455     
PER                    0.9091     0.9821  0.9442    112     
REGULATION             0.9118     0.8378  0.8732    37      
REPORT                 0.7778     0.7778  0.7778    9       
TREATY                 1.0000     1.0000  1.0000    3       
micro-avg              0.9139     0.9116  0.9127    1222    
macro-avg              0.9089     0.9133  0.9024    1222    
weighted-avg           0.9143     0.9116  0.9124    1222
```
