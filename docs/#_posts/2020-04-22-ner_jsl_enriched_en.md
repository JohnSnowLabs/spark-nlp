---
layout: model
title: Detect Diagnosis, Symptoms, Drugs, Labs and Demographics (ner_jsl_enriched)
author: John Snow Labs
name: ner_jsl_enriched_en
date: 2020-04-22
task: Named Entity Recognition
language: en
edition: Healthcare NLP 2.4.2
spark_version: 2.4
tags: [ner, en, clinical, licensed]
supported: true
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description


Pretrained named entity recognition deep learning model for clinical terminology. The SparkNLP deep learning model (NerDL) is inspired by a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM-CNN. 


## Predicted Entities 
`Age`, `Diagnosis`, `Dosage`, `Drug_Name`, `Frequency`, `Gender`, `Lab_Name`, `Lab_Result`, `Symptom_Name`, `Allergenic_substance`, `Blood_Pressure`, `Causative_Agents_(Virus_and_Bacteria)`, `Modifier`, `Name`, `Negation`, `O2_Saturation`, `Procedure`, `Procedure_Name`, `Pulse_Rate`, `Respiratory_Rate`, `Route`, `Section_Name`, `Substance_Name`, `Temperature`, `Weight`.


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_JSL/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_JSL.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_jsl_enriched_en_2.4.2_2.4_1587513303751.zip){:.button.button-orange.button-orange-trans.arr.button-icon}




## How to use
Use as part of an nlp pipeline with the following stages: DocumentAssembler, SentenceDetector, Tokenizer, WordEmbeddingsModel, NerDLModel. Add the NerConverter to the end of the pipeline to convert entity tokens into full entity chunks.


<div class="tabs-box" markdown="1">


{% include programmingLanguageSelectScalaPython.html %}




```python
...
word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
  .setInputCols(["sentence", "token"])\
  .setOutputCol("embeddings")
  
clinical_ner = NerDLModel.pretrained("ner_jsl_enriched", "en", "clinical/models") \
  .setInputCols(["sentence", "token", "embeddings"]) \
  .setOutputCol("ner")
...

nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter])
model = nlpPipeline.fit(spark.createDataFrame([[""]]).toDF("text"))


results = model.transform(spark.createDataFrame([["The patient is a 21-day-old Caucasian male here for 2 days of congestion - mom has been suctioning yellow discharge from the patient's nares, plus she has noticed some mild problems with his breathing while feeding (but negative for any perioral cyanosis or retractions). One day ago, mom also noticed a tactile temperature and gave the patient Tylenol. Baby also has had some decreased p.o. intake. His normal breast-feeding is down from 20 minutes q.2h. to 5 to 10 minutes secondary to his respiratory congestion. He sleeps well, but has been more tired and has been fussy over the past 2 days. The parents noticed no improvement with albuterol treatments given in the ER. His urine output has also decreased; normally he has 8 to 10 wet and 5 dirty diapers per 24 hours, now he has down to 4 wet diapers per 24 hours. Mom denies any diarrhea. His bowel movements are yellow colored and soft in nature."]], ["text"]))
```


```scala
...
val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
  .setInputCols(Array("sentence", "token"))
  .setOutputCol("embeddings")
val ner = NerDLModel.pretrained("ner_jsl_enriched", "en", "clinical/models")
  .setInputCols("sentence", "token", "embeddings")
  .setOutputCol("ner")
...
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, ner, ner_converter))


val data = Seq("The patient is a 21-day-old Caucasian male here for 2 days of congestion - mom has been suctioning yellow discharge from the patient's nares, plus she has noticed some mild problems with his breathing while feeding (but negative for any perioral cyanosis or retractions). One day ago, mom also noticed a tactile temperature and gave the patient Tylenol. Baby also has had some decreased p.o. intake. His normal breast-feeding is down from 20 minutes q.2h. to 5 to 10 minutes secondary to his respiratory congestion. He sleeps well, but has been more tired and has been fussy over the past 2 days. The parents noticed no improvement with albuterol treatments given in the ER. His urine output has also decreased; normally he has 8 to 10 wet and 5 dirty diapers per 24 hours, now he has down to 4 wet diapers per 24 hours. Mom denies any diarrhea. His bowel movements are yellow colored and soft in nature.").toDF("text")

val result = pipeline.fit(data).transform(data)
```


</div>


{:.h2_title}
## Results
The output is a dataframe with a sentence per row and a ``"ner"`` column containing all of the entity labels in the sentence, entity character indices, and other metadata. To get only the tokens and entity labels, without the metadata, select ``"token.result"`` and ``"ner.result"`` from your output dataframe or add the ``"Finisher"`` to the end of your pipeline.


```bash
+---------------------------+------------+
|chunk                      |ner         |
+---------------------------+------------+
|21-day-old                 |Age         |
|male                       |Gender      |
|congestion                 |Symptom_Name|
|mom                        |Gender      |
|suctioning yellow discharge|Symptom_Name|
|she                        |Gender      |
|problems with his breathing|Symptom_Name|
|perioral cyanosis          |Symptom_Name|
|retractions                |Symptom_Name|
|mom                        |Gender      |
|Tylenol                    |Drug_Name   |
|His                        |Gender      |
|his                        |Gender      |
|respiratory congestion     |Symptom_Name|
|He                         |Gender      |
|tired                      |Symptom_Name|
|fussy                      |Symptom_Name|
|albuterol                  |Drug_Name   |
+---------------------------+------------+
```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|ner_jsl_enriched_en_2.4.2_2.4|
|Type:|ner|
|Compatibility:|Spark NLP 2.4.2|
|Edition:|Official|
|License:|Licensed|
|Input Labels:|[sentence,token, embeddings]|
|Output Labels:|[ner]|
|Language:|[en]|
|Case sensitive:|false|


{:.h2_title}
## Data Source
Trained on data gathered and manually annotated by John Snow Labs.
https://www.johnsnowlabs.com/data/


{:.h2_title}
## Benchmarking
```bash
label                                       tp     fp     fn      prec       rec        f1
B-Pulse_Rate                                80     26      9  0.754717  0.898876  0.820513
I-Diagnosis                               2341   1644   1129  0.587453  0.67464   0.628035
I-Procedure_Name                          2209   1128   1085  0.661972  0.670613  0.666265
B-Lab_Result                               432    107    263  0.801484  0.621583  0.700162
B-Dosage                                   465    179     81  0.72205   0.851648  0.781513
I-Causative_Agents_(Virus_and_Bacteria)      9      3     10  0.75      0.473684  0.580645
B-Name                                     648    295    510  0.687169  0.559585  0.616849
I-Name                                     917    427    665  0.682292  0.579646  0.626794
B-Weight                                    52     25      9  0.675325  0.852459  0.753623
B-Symptom_Name                            4244   1911   1776  0.689521  0.704983  0.697166
I-Maybe                                     25     15     63  0.625     0.284091  0.390625
I-Symptom_Name                            1920   1584   2503  0.547945  0.434095  0.48442 
B-Modifier                                1399    704    942  0.66524   0.597608  0.629613
B-Blood_Pressure                            82     21      7  0.796117  0.921348  0.854167
B-Frequency                                290     93     97  0.75718   0.749354  0.753247
I-Gender                                    29     19     25  0.604167  0.537037  0.568627
I-Age                                        3      6     11  0.333333  0.214286  0.26087 
B-Drug_Name                               1762    500    271  0.778957  0.866699  0.820489
B-Substance_Name                           143     32     53  0.817143  0.729592  0.770889
B-Temperature                               58     23     11  0.716049  0.84058   0.773333
B-Section_Name                            2700    294    177  0.901804  0.938478  0.919775
I-Route                                    131    165    177  0.442568  0.425325  0.433775
B-Maybe                                    108     47    164  0.696774  0.397059  0.505855
B-Gender                                  5156    685     68  0.882726  0.986983  0.931948
I-Dosage                                   435    182     87  0.705024  0.833333  0.763828
B-Causative_Agents_(Virus_and_Bacteria)     21     17      6  0.552632  0.777778  0.646154
I-Frequency                                278    131    191  0.679707  0.592751  0.633257
B-Age                                      352     34     21  0.911917  0.9437    0.927536
I-Lab_Result                                27     20    170  0.574468  0.137056  0.221311
B-Negation                                1501    311    341  0.828366  0.814875  0.821565
B-Diagnosis                               2657   1281   1049  0.674708  0.716945  0.695186
I-Section_Name                            3876   1304    188  0.748263  0.95374   0.838598
B-Route                                    466    286    123  0.619681  0.791172  0.695004
I-Negation                                  80    152    190  0.344828  0.296296  0.318725
B-Procedure_Name                          1453    739    562  0.662865  0.721092  0.690754
I-Allergenic_substance                       6      1      7  0.857143  0.461538  0.6     
B-Allergenic_substance                      74     31     23  0.704762  0.762887  0.732673
I-Weight                                    46     43     17  0.516854  0.730159  0.605263
B-Lab_Name                                 639    189    287  0.771739  0.690065  0.72862 
I-Modifier                                 104    156    417  0.4       0.199616  0.266325
I-Temperature                                2      7     13  0.222222  0.133333  0.166667
I-Drug_Name                                334    237    290  0.584939  0.535256  0.558996
I-Lab_Name                                 271    157    140  0.633178  0.659367  0.646007
B-Respiratory_Rate                          46      6      5  0.884615  0.901961  0.893204
Macro-average                            37896  15237  14343  0.621144  0.562248  0.59023 
Micro-average                            37896  15237  14343  0.713229  0.725435  0.71928 
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbNzEzNTgzNTk1LDEwMTg5NTAyMDUsMjA1MD
M0MTA4Nl19
-->