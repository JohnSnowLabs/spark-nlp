---
layout: model
title: Legal NER for Indian Court Documents
author: John Snow Labs
name: legner_indian_court_judgement
date: 2022-10-25
tags: [en, legal, ner, licensed]
task: Named Entity Recognition
language: en
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
annotator: LegalNerModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is an NER model trained on Indian court dataset, aimed to extract the following entities from judgement documents.

## Predicted Entities

`COURT`, `PETITIONER`, `RESPONDENT`, `JUDGE`, `DATE`, `ORG`, `GPE`, `STATUTE`, `PROVISION`, `PRECEDENT`, `CASE_NUMBER`, `WITNESS`, `OTHER_PERSON`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legner_indian_court_judgement_en_1.0.0_3.0_1666698501448.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = nlp.DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")\
    .setCleanupMode("shrink")

sentence_detector = nlp.SentenceDetectorDLModel.pretrained("sentence_detector_dl", "en")\
    .setInputCols(["document"])\
    .setOutputCol("sentence")\

tokenizer = nlp.Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

embeddings = nlp.BertEmbeddings.pretrained("bert_base_cased", "en")\
    .setInputCols("sentence", "token")\
    .setOutputCol("embeddings")\
    .setMaxSentenceLength(512)\
    .setCaseSensitive(True)

ner_model = legal.NerModel.pretrained("legner_indian_court_judgement", "en", "legal/models")\
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

data = spark.createDataFrame([["""Let fresh bailable warrant of Rs.20,000/- (Rupees Twenty Thousand) be issued through Superintendent of Police, Dhar to the respondents No.1 Sikandar and No.2 Aziz for a date to be fixed by the Registry to secure the presence of the respondents No.1 and 2, made returnable within six weeks.
P.K.Jaiswal)  Judge                  
(Jarat Kumar Jain) Judge ns.
W.P.No.1361/2013 
14/12/2015              
Parties through their Counsel."""]])
                             
result = pipeline.fit(data).transform(data)
```
```scala
val document_assembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")
    .setCleanupMode("shrink")

val sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "en")
    .setInputCols(Array("document"))
    .setOutputCol("sentence")

val tokenizer = new Tokenizer()
    .setInputCols(Array("sentence"))
    .setOutputCol("token")

val embeddings = BertEmbeddings.pretrained("bert_base_cased", "en")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("embeddings")
    .setMaxSentenceLength(512)
    .setCaseSensitive(True)

val ner_model = NerModel.pretrained("legner_indian_court_judgement", "en", "legal/models")
    .setInputCols(Array("sentence", "token", "embeddings"))
    .setOutputCol("ner")

val ner_converter = new NerConverter()
    .setInputCols(Array("sentence", "token", "ner"))
    .setOutputCol("ner_chunk")

val pipeline = new Pipeline().setStages(Array(
    document_assembler,
    sentence_detector,
    tokenizer,
    embeddings,
    ner_model,
    ner_converter))

val data = Seq("""Let fresh bailable warrant of Rs.20,000/- (Rupees Twenty Thousand) be issued through Superintendent of Police, Dhar to the respondents No.1 Sikandar and No.2 Aziz for a date to be fixed by the Registry to secure the presence of the respondents No.1 and 2, made returnable within six weeks.
P.K.Jaiswal)  Judge                  
(Jarat Kumar Jain) Judge ns.
W.P.No.1361/2013 
14/12/2015              
Parties through their Counsel.""").toDS.toDF("text")
                             
val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+----------------+-----------+
|chunk           |label      |
+----------------+-----------+
|Dhar            |GPE        |
|Sikandar        |RESPONDENT |
|Aziz            |RESPONDENT |
|P.K.Jaiswal     |JUDGE      |
|Jarat Kumar Jain|JUDGE      |
|W.P.No.1361/2013|CASE_NUMBER|
|14/12/2015      |DATE       |
+----------------+-----------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legner_indian_court_judgement|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|16.4 MB|

## References

Training data is available [here](https://github.com/Legal-NLP-EkStep/legal_NER#3-data).

## Benchmarking

```bash
label         precision  recall  f1-score  support 
CASE_NUMBER   0.83       0.80    0.82      112     
COURT         0.92       0.94    0.93      140     
DATE          0.97       0.97    0.97      204     
GPE           0.81       0.75    0.78      95      
JUDGE         0.84       0.86    0.85      57      
ORG           0.75       0.76    0.76      131     
OTHER_PERSON  0.83       0.90    0.86      241     
PETITIONER    0.76       0.61    0.68      36      
PRECEDENT     0.84       0.84    0.84      127     
PROVISION     0.90       0.94    0.92      220     
RESPONDENT    0.64       0.70    0.67      23      
STATUTE       0.92       0.96    0.94      157     
WITNESS       0.93       0.78    0.85      87      
micro-avg     0.87       0.87    0.87      1630    
macro-avg     0.84       0.83    0.83      1630    
weighted-avg  0.87       0.87    0.87      1630
```
