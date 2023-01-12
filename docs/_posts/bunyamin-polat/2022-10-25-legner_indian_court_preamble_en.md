---
layout: model
title: Legal NER for Indian Court Documents
author: John Snow Labs
name: legner_indian_court_preamble
date: 2022-10-25
tags: [en, licensed, legal, ner]
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

This is an NER model trained on Indian court dataset, aimed to extract the following entities from preamble documents.

## Predicted Entities

`COURT`, `PETITIONER`, `RESPONDENT`, `JUDGE`, `LAWYER`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legner_indian_court_preamble_en_1.0.0_3.0_1666702718567.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
    .setOutputCol("sentence")

tokenizer = nlp.Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

embeddings = nlp.BertEmbeddings.pretrained("bert_base_cased", "en")\
    .setInputCols("sentence", "token")\
    .setOutputCol("embeddings")\
    .setMaxSentenceLength(512)\
    .setCaseSensitive(True)

ner_model = legal.NerModel.pretrained("legner_indian_court_preamble", "en", "legal/models")\
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


data = spark.createDataFrame([["""In The High Court Of Judicature At Madras 

                Dated:  31/05/2006  
                
                
The Hon'Ble Mr. Justice V. Dhanapalan         
                
C.M.A.No.535 of 1998

1. Sahabudeen               
             ...     Claimant/Appellant
                            
Vs


1. R. Selvaraj,

2. The New India Assurance Co.Ltd., 

                        ...     Respondents


Appeal filed under Section 173 of the Motor Vehicles Act to set  aside
the  judgment  and decree dated 25.03.97 passed in Mcop No.5/95 on the file of
the I Additional District Judge-cum-Chief Judicial Magistrate, Coimbatore  and
pass  the  award  of  Rs.3,50,000/-  instead  of  Rs.1,00  ,000/-  towards the
compensation to the petitioner.


For Petitioner :  Mr.  K.Sudarsanam for M/s.  Surithi Associates

For Respondents:  Mr.  Mohd.Fiary Hussain for R1"""]])
                             
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

val ner_model = NerModel.pretrained("legner_indian_court_preamble", "en", "legal/models")
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

val data = Seq("""In The High Court Of Judicature At Madras 

                Dated:  31/05/2006  
                
                
The Hon'Ble Mr. Justice V. Dhanapalan         
                
C.M.A.No.535 of 1998

1. Sahabudeen               
             ...     Claimant/Appellant
                            
Vs


1. R. Selvaraj,

2. The New India Assurance Co.Ltd., 

                        ...     Respondents


Appeal filed under Section 173 of the Motor Vehicles Act to set  aside
the  judgment  and decree dated 25.03.97 passed in Mcop No.5/95 on the file of
the I Additional District Judge-cum-Chief Judicial Magistrate, Coimbatore  and
pass  the  award  of  Rs.3,50,000/-  instead  of  Rs.1,00  ,000/-  towards the
compensation to the petitioner.


For Petitioner :  Mr.  K.Sudarsanam for M/s.  Surithi Associates

For Respondents:  Mr.  Mohd.Fiary Hussain for R1""").toDS.toDF("text")
                             
val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+----------------------------------+----------+
|chunk                             |label     |
+----------------------------------+----------+
|High Court Of Judicature At Madras|COURT     |
|V. Dhanapalan                     |JUDGE     |
|Sahabudeen                        |PETITIONER|
|Selvaraj                          |RESPONDENT|
|New India Assurance               |RESPONDENT|
|K.Sudarsanam                      |LAWYER    |
|Mohd.Fiary Hussain                |LAWYER    |
+----------------------------------+----------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legner_indian_court_preamble|
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
COURT         0.92       0.91    0.91      109     
JUDGE         0.96       0.92    0.94      168     
LAWYER        0.94       0.93    0.94      377     
PETITIONER    0.76       0.77    0.76      269     
RESPONDENT    0.78       0.80    0.79      356     
micro-avg     0.86       0.86    0.86      1279    
macro-avg     0.87       0.86    0.87      1279    
weighted-avg  0.86       0.86    0.86      1279
```
