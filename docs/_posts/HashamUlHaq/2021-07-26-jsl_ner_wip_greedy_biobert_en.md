---
layout: model
title: Detect Clinical Entities (jsl_ner_wip_greedy_biobert)
author: John Snow Labs
name: jsl_ner_wip_greedy_biobert
date: 2021-07-26
tags: [ner, licensed, clinical, en]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 3.1.3
spark_version: 3.0
supported: true
annotator: MedicalNerModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description


Pretrained named entity recognition deep learning model for clinical terminology. The SparkNLP deep learning model (MedicalNerModel) is inspired by a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM-CNN.


## Predicted Entities


`Test_Result`, `Relationship_Status`, `RelativeDate`, `Blood_Pressure`, `Triglycerides`, `Smoking`, `Pregnancy`, `Medical_History_Header`, `LDL`, `Hypertension`, `Hyperlipidemia`, `Frequency`, `BMI`, `Internal_organ_or_component`, `Allergen`, `Fetus_NewBorn`, `Substance_Quantity`, `Time`, `Temperature`, `Procedure`, `Strength`, `Treatment`, `HDL`, `Alcohol`, `Birth_Entity`, `Diet`, `Weight`, `Oxygen_Therapy`, `Injury_or_Poisoning`, `Section_Header`, `Obesity`, `EKG_Findings`, `Gender`, `Height`, `Social_History_Header`, `Diabetes`, `Route`, `Race_Ethnicity`, `Substance`, `Drug`, `External_body_part_or_region`, `RelativeTime`, `Admission_Discharge`, `Psychological_Condition`, `Total_Cholesterol`, `Labour_Delivery`, `Imaging_Technique`, `Date`, `Form`, `Overweight`, `Cerebrovascular_Disease`, `Vital_Signs_Header`, `Oncological`, `ImagingFindings`, `Communicable_Disease`, `Duration`, `Vaccine`, `Kidney_Disease`, `O2_Saturation`, `Heart_Disease`, `Employment`, `Sexually_Active_or_Sexual_Orientation`, `Test`, `Disease_Syndrome_Disorder`, `Respiration`, `Direction`, `Medical_Device`, `Clinical_Dept`, `Modifier`, `Symptom`, `Pulse`, `Age`, `Death_Entity`, `Dosage`, `Family_History_Header`, `VS_Finding`


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_JSL/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_JSL.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/jsl_ner_wip_greedy_biobert_en_3.1.3_3.0_1627304288213.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


## How to use






<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = DocumentAssembler()\
		.setInputCol("text")\
		.setOutputCol("document")

sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare", "en", "clinical/models") \
		.setInputCols(["document"]) \
		.setOutputCol("sentence")

tokenizer = Tokenizer()\
		.setInputCols(["sentence"])\
		.setOutputCol("token")
	
embeddings = BertEmbeddings.pretrained("biobert_pubmed_base_cased")\
		.setInputCols(["sentence",  "token"]) \
		.setOutputCol("embeddings")
		
jsl_ner = MedicalNerModel.pretrained("jsl_ner_wip_greedy_biobert", "en", "clinical/models") \
		.setInputCols(["sentence", "token", "embeddings"]) \
		.setOutputCol("jsl_ner")

jsl_ner_converter = NerConverter() \
		.setInputCols(["sentence", "token", "jsl_ner"]) \
		.setOutputCol("ner_chunk")

jsl_ner_pipeline = Pipeline().setStages([
				documentAssembler,
				sentenceDetector,
				tokenizer,
				embeddings,
				jsl_ner,
				jsl_ner_converter])


jsl_ner_model = jsl_ner_pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

data = spark.createDataFrame([["""The patient is a 21-day-old Caucasian male here for 2 days of congestion - mom has been suctioning yellow discharge from the patient's nares, plus she has noticed some mild problems with his breathing while feeding (but negative for any perioral cyanosis or retractions). One day ago, mom also noticed a tactile temperature and gave the patient Tylenol. Baby also has had some decreased p.o. intake. His normal breast-feeding is down from 20 minutes q.2h. to 5 to 10 minutes secondary to his respiratory congestion. He sleeps well, but has been more tired and has been fussy over the past 2 days. The parents noticed no improvement with albuterol treatments given in the ER. His urine output has also decreased; normally he has 8 to 10 wet and 5 dirty diapers per 24 hours, now he has down to 4 wet diapers per 24 hours. Mom denies any diarrhea. His bowel movements are yellow colored and soft in nature."""]]).toDF("text")

result = jsl_ner_model.transform(data)
```
```scala
val documentAssembler = new DocumentAssembler()
		.setInputCol("text")
		.setOutputCol("document")

val sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare", "en", "clinical/models")
		.setInputCols("document") 
		.setOutputCol("sentence")

val tokenizer = new Tokenizer()
		.setInputCols("sentence")
		.setOutputCol("token")
	
val embeddings = BertEmbeddings.pretrained("biobert_pubmed_base_cased")
		.setInputCols(Array("sentence",  "token")) 
		.setOutputCol("embeddings")
  
val jsl_ner = MedicalNerModel.pretrained("jsl_ner_wip_greedy_biobert", "en", "clinical/models")
		.setInputCols(Array("sentence", "token", "embeddings"))
		.setOutputCol("jsl_ner")

val jsl_ner_converter = new NerConverter()
		.setInputCols(Array("sentence", "token", "jsl_ner"))
		.setOutputCol("ner_chunk")
 
val jsl_ner_pipeline = new Pipeline().setStages(Array(
					documentAssembler, 
					sentenceDetector, 
					tokenizer, 
					embeddings, 
					jsl_ner, 
					jsl_ner_converter))


val data = Seq("""The patient is a 21-day-old Caucasian male here for 2 days of congestion - mom has been suctioning yellow discharge from the patient's nares, plus she has noticed some mild problems with his breathing while feeding (but negative for any perioral cyanosis or retractions). One day ago, mom also noticed a tactile temperature and gave the patient Tylenol. Baby also has had some decreased p.o. intake. His normal breast-feeding is down from 20 minutes q.2h. to 5 to 10 minutes secondary to his respiratory congestion. He sleeps well, but has been more tired and has been fussy over the past 2 days. The parents noticed no improvement with albuterol treatments given in the ER. His urine output has also decreased; normally he has 8 to 10 wet and 5 dirty diapers per 24 hours, now he has down to 4 wet diapers per 24 hours. Mom denies any diarrhea. His bowel movements are yellow colored and soft in nature.""").toDS.toDF("text")

val result = jsl_ner_pipeline.fit(data).transform(data)
```
</div>


## Results


```bash
|    | chunk                                          | entity                       |
|---:|:-----------------------------------------------|:-----------------------------|
|  0 | 21-day-old                                     | Age                          |
|  1 | Caucasian                                      | Race_Ethnicity               |
|  2 | male                                           | Gender                       |
|  3 | for 2 days                                     | Duration                     |
|  4 | congestion                                     | Symptom                      |
|  5 | mom                                            | Gender                       |
|  6 | suctioning yellow discharge                    | Symptom                      |
|  7 | nares                                          | External_body_part_or_region |
|  8 | she                                            | Gender                       |
|  9 | mild problems with his breathing while feeding | Symptom                      |
| 10 | perioral cyanosis                              | Symptom                      |
| 11 | retractions                                    | Symptom                      |
| 12 | One day ago                                    | RelativeDate                 |
| 13 | mom                                            | Gender                       |
| 14 | tactile temperature                            | Symptom                      |
| 15 | Tylenol                                        | Drug                         |
| 16 | Baby                                           | Age                          |
| 17 | decreased p.o. intake                          | Symptom                      |
| 18 | His                                            | Gender                       |
| 19 | breast-feeding                                 | External_body_part_or_region |
| 20 | q.2h                                           | Frequency                    |
| 21 | to 5 to 10 minutes                             | Duration                     |
| 22 | his                                            | Gender                       |
| 23 | respiratory congestion                         | Symptom                      |
| 24 | He                                             | Gender                       |
| 25 | tired                                          | Symptom                      |
| 26 | fussy                                          | Symptom                      |
| 27 | over the past 2 days                           | RelativeDate                 |
| 28 | albuterol                                      | Drug                         |
| 29 | ER                                             | Clinical_Dept                |
| 30 | His                                            | Gender                       |
| 31 | urine output has also decreased                | Symptom                      |
| 32 | he                                             | Gender                       |
| 33 | per 24 hours                                   | Frequency                    |
| 34 | he                                             | Gender                       |
| 35 | per 24 hours                                   | Frequency                    |
| 36 | Mom                                            | Gender                       |
| 37 | diarrhea                                       | Symptom                      |
| 38 | His                                            | Gender                       |
| 39 | bowel                                          | Internal_organ_or_component  |
```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|jsl_ner_wip_greedy_biobert|
|Compatibility:|Healthcare NLP 3.1.3+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|


## Data Source


Trained on data gathered and manually annotated by John Snow Labs.
https://www.johnsnowlabs.com/data/


## Benchmarking


```bash
label                                    tp	    fp	   fn	    prec	     rec	       f1
B-Oxygen_Therapy                         47	    11	   10	    0.8103448	 0.8245614	 0.81739134
B-Cerebrovascular_Disease                43	    20	   21	    0.6825397	 0.671875	   0.6771653
B-Triglycerides                          5      0      0	    1.0	       1.0	       1.0
I-Cerebrovascular_Disease                25	    12     27     0.6756757	 0.48076922	 0.56179774
B-Medical_Device                         2704   531    364    0.8358578	 0.88135594	 0.85800415
B-Labour_Delivery                        43	    16	   29	    0.7288136	 0.5972222	 0.6564886
I-Vaccine                                5	     0      5	    1.0	       0.5	       0.6666667
I-Obesity                                6	     4      1	    0.6	       0.85714287	 0.70588243
I-Smoking                                3	     1      2	    0.75	     0.6	       0.6666667
B-RelativeTime	                         67	    36	   51	    0.65048546 0.5677966	 0.60633487
B-Imaging_Technique	                     33	    12	   19	    0.73333335 0.63461536	 0.68041235
B-Heart_Disease	                         285    55	   68	    0.8382353	 0.8073654	 0.82251084
B-Procedure	                             1876   303	   384    0.8609454	 0.8300885	 0.84523547
I-RelativeTime	                         105    43	   53	    0.7094595	 0.664557	   0.6862745
B-Drug	                                 1803   299	   265    0.8577545	 0.87185687	 0.8647482
B-Obesity                                29	    9	     5	    0.7631579	 0.85294116	 0.8055555
I-RelativeDate	                         617    167	   107    0.7869898	 0.8522099	 0.8183024
B-O2_Saturation	                         27	    8	     6	    0.7714286	 0.8181818	 0.7941177
B-Direction	                             2856   390	   326    0.8798521	 0.89754874	 0.88861233
I-Alcohol                                4      4	     4	    0.5	       0.5	       0.5
I-Oxygen_Therapy                         25	    7	     6	    0.78125	   0.8064516	 0.79365087
B-Diet                                   23	    14	   32	    0.6216216	 0.4181818	 0.5
B-Dosage                                 35	    26	   29	    0.57377046 0.546875	   0.55999994
B-Injury_or_Poisoning                    308    52	   83	    0.85555553 0.7877238	 0.82023966
B-Hypertension	                         80	    9      2	    0.8988764	 0.9756098	 0.9356726
I-Test_Result                            124    73     156    0.6294416	 0.44285715	 0.5199161
B-Alcohol                                54	    11	   12	    0.830	     0.8181818	 0.8244275
B-Height                                 14	    5	     5	    0.7368421	 0.7368421	 0.7368421
I-Substance	                             18	    8	     8	    0.6923077	 0.6923077	 0.6923077
B-RelativeDate	                         372    109	   93	    0.7733888	 0.8	       0.78646934
B-Admission_Discharge                    218    22	   14	    0.90833336 0.9396552	 0.9237288
B-Date	                                 345    24	   26	    0.93495935 0.9299191	 0.9324324
B-Kidney_Disease                         63	    10	   20	    0.8630137	 0.7590361	 0.8076923
I-Strength	                             22	    17	   13	    0.5641026	 0.62857145	 0.59459466
I-Injury_or_Poisoning                    301    93	   98	    0.7639594	 0.75438595	 0.75914246
I-Time	                                 28	    11	   17	    0.71794873 0.62222224	 0.6666667
B-Substance	                             48	    11	   10	    0.8135593	 0.82758623	 0.8205129
B-Total_Cholesterol	                     6      3	     0	    0.6666667	 1.0	       0.8
I-Vital_Signs_Header                     276    28	   8	    0.90789473 0.97183096	 0.93877554
I-Internal_organ_or_component            2907   518	   490    0.8487591	 0.8557551	 0.8522427
B-Hyperlipidemia                         28	    3      0	    0.9032258	 1.0	       0.9491525
B-Overweight                             3      0      3	    1.0	       0.5	       0.6666667
I-Sexually_Active_or_Sexual_Orientation  2      0      3	    1.0	       0.4	       0.5714286
B-Sexually_Active_or_Sexual_Orientation  2      0      2	    1.0	       0.5	       0.6666667
I-Fetus_NewBorn	                         50	    38	   58	    0.5681818	 0.46296296	 0.5102041
B-BMI                                    6      0      1	    1.0	       0.85714287	 0.9230769
B-ImagingFindings                        52     41	   61	    0.5591398	 0.460177	   0.5048544
B-Test_Result                            714    135	   212    0.8409894	 0.7710583	 0.8045071
B-Section_Header                         2140   79	   65	    0.9643984	 0.97052157	 0.96745026
I-Treatment	                             85	    21	   29	    0.8018868	 0.74561405	 0.7727273
B-Clinical_Dept	                         638    82	   77	    0.88611114 0.8923077	 0.88919866
I-Kidney_Disease                         114    7      18	    0.94214875 0.8636364	 0.90118575
I-Pulse	                                 189    27	   42	    0.875	     0.8181818	 0.84563756
B-Test	                                 1589   320	   315    0.83237296 0.83455884	 0.83346444
B-Weight                                 54	    12	   13	    0.8181818	 0.80597013	 0.81203
I-Respiration                            114    4      17	    0.9661017	 0.870229	   0.91566265
I-EKG_Findings	                         68	    34	   52	    0.6666667	 0.56666666	 0.6126126
I-Section_Header                         3828   168	   77	    0.957958	 0.9802817	 0.9689913
B-Strength	                             27	    13	   23	    0.675	     0.54	       0.6
I-Social_History_Header	                 137    4      4	    0.9716312	 0.9716312	 0.9716312
B-Vital_Signs_Header                     183    18	   7	    0.9104478	 0.9631579	 0.9360614
B-Death_Entity	                         28	    9      6	    0.7567568	 0.8235294	 0.7887324
B-Modifier	                             302    90	   282    0.77040815 0.5171233	 0.6188525
B-Blood_Pressure                         93	    14	   21	    0.86915886 0.81578946	 0.84162897
I-O2_Saturation	                         49	    19	   23	    0.7205882	 0.6805556	 0.7
B-Frequency	                             437    77	   68	    0.8501946	 0.86534655	 0.8577036
I-Triglycerides	                         5      0      0	    1.0	       1.0	       1.0
I-Duration	                             513    254	   47	    0.66883963 0.9160714	 0.77317256
I-Diabetes	                             50	    4      6	    0.9259259	 0.89285713	 0.90909094
B-Race_Ethnicity                         78	    3      2	    0.962963	 0.975	     0.9689441
I-Gender                                 114    2      17	    0.98275864 0.870229	   0.9230769
I-Height                                 43	    13	   10	    0.76785713 0.8113208	 0.78899086
B-Communicable_Disease	                 10	    5      9	    0.6666667	 0.5263158	 0.5882354
I-Family_History_Header	                 134    1      0	    0.9925926	 1.0	       0.9962825
B-LDL                                    2      2      2	    0.5	       0.5	       0.5
I-Race_Ethnicity                         6      0      0	    1.0	       1.0	       1.0
B-Psychological_Condition                103    21	   17	    0.83064514 0.85833335	 0.84426236
I-Age                                    116    14	   50	    0.8923077	 0.6987952	 0.78378385
B-EKG_Findings	                         33	    18	   32	    0.64705884 0.50769234	 0.56896555
B-Employment                             168    29	   44	    0.8527919	 0.7924528	 0.8215159
I-Oncological                            358    38	   17	    0.9040404	 0.9546667	 0.9286641
B-Time	                                 27	    7      18	    0.7941176	 0.6	       0.68354434
B-Treatment	                             93	    31     41	    0.75	     0.69402987	 0.7209303
B-Temperature                            69	    5      8	    0.9324324	 0.8961039	 0.9139073
I-Procedure	                             2437   379    501    0.86541194 0.8294758	 0.84706295
B-Relationship_Status                    30	    3      1	    0.90909094 0.9677419	 0.9375
B-Pregnancy	                             56	    17	   30	    0.7671233	 0.6511628	 0.7044025
I-Route	                                 8      4      7	    0.6666667	 0.53333336	 0.59259266
I-Medical_History_Header                 151    4      15	    0.9741936	 0.9096386	 0.94080997
I-Imaging_Technique	                     25	    5      20	    0.8333333	 0.5555556	 0.66666675
B-Smoking                                74	    6      4	    0.925	     0.94871795	 0.93670887
I-Labour_Delivery                        36	    8      18	    0.8181818	 0.6666667	 0.7346939
I-Death_Entity	                         3      0      2	    1.0	       0.6	       0.75
B-Diabetes	                             77	    9      5	    0.89534885 0.9390244	 0.9166666
B-Gender                                 4479   82	   111    0.9820215	 0.97581697	 0.9789094
B-Vaccine                                6      1      9	    0.85714287 0.4	       0.54545456
I-Heart_Disease	                         393    61	   89	    0.8656388	 0.8153527	 0.8397436
I-Dosage                                 31	    27	   22	    0.5344828	 0.5849057	 0.5585586
B-Social_History_Header	                 78	    2      3	    0.975	     0.962963	   0.9689441
B-External_body_part_or_region	         1640   402	   311    0.8031342	 0.8405946	 0.8214376
I-Clinical_Dept	                         546    59	   47	    0.90247935 0.920742	   0.91151917
I-Test	                                 1195   320	   402    0.7887789	 0.748278	   0.7679949
I-Frequency	                             340    97	   120    0.77803206 0.73913044	 0.75808245
B-Age                                    454    35	   57	    0.9284254	 0.888454	   0.908
B-Pulse	                                 90	    11	   17	    0.8910891	 0.8411215	 0.8653846
I-Symptom                                4265   2050   1232   0.6753761	 0.7758778	 0.72214705
I-Pregnancy	                             39	    28	   42	    0.58208954 0.4814815	 0.527027
I-LDL                                    5      0      4	    1.0	       0.5555556	 0.71428573
I-Diet	                                 33	    14	   25	    0.70212764 0.5689655	 0.6285714
I-Blood_Pressure                         198    54	   27	    0.78571427 0.88	       0.83018863
I-ImagingFindings                        136    99	   85	    0.57872343 0.61538464	 0.5964913
I-Date	                                 203    13	   10	    0.9398148	 0.9530516	 0.946387
B-Route	                                 84	    23	   47	    0.78504676 0.64122134	 0.7058824
B-Duration	                             204    110	   26	    0.6496815	 0.8869565	 0.74999994
B-Medical_History_Header                 56	    1	     7	    0.98245615 0.8888889	 0.93333334
B-Respiration                            55	    4	     6	    0.9322034	 0.90163934	 0.9166667
I-External_body_part_or_region	         314    105	   167    0.74940336 0.65280664	 0.6977778
I-BMI                                    15	    0	     1	    1.0	       0.9375	     0.9677419
B-Internal_organ_or_component            4349   886	   761    0.8307545	 0.8510763	 0.8407926
I-Weight                                 150    22	   23	    0.872093	 0.867052	   0.8695652
B-Disease_Syndrome_Disorder	             1698   375	   358    0.81910276 0.82587546	 0.8224752
B-Symptom                                4358   1002   932    0.8130597	 0.8238185	 0.8184037
B-VS_Finding                             138    36	   37	    0.79310346 0.7885714	 0.79083097
I-Disease_Syndrome_Disorder	             1723   372	   451    0.82243437 0.7925483	 0.8072148
I-Drug	                                 3282   838	   493    0.79660195 0.86940396	 0.8314123
I-Medical_Device                         1864   418	   242    0.81682736 0.88509023	 0.84958977
B-Oncological                            278    22	   22	    0.9266667	 0.9266667	 0.9266667
I-Temperature                            111    8      6	    0.9327731	 0.94871795	 0.94067794
I-Employment                             92	    27	   19	    0.77310926 0.8288288	 0.8
I-Psychological_Condition                32	    7      19	    0.82051283 0.627451	   0.7111111
B-Family_History_Header	                 68	    0      0	    1.0	       1.0	       1.0
I-Direction	                             311    91	   144    0.7736318	 0.6835165	 0.72578764
Macro-average                            65035  12855  11898  0.761429   0.70630085  0.7328297
Micro-average                            65035  12855  11898  0.83495957 0.845346    0.8401207
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE4NjI5NjcyMTUsLTMzNzU0NjQ4NywyND
cwNDY5NzcsMTYzOTYzNDEyMSwxMjMyOTI2NzgyXX0=
-->
