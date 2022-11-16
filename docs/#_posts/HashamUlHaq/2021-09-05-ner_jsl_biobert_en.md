---
layout: model
title: Detect Clinical Entities (ner_jsl_biobert)
author: John Snow Labs
name: ner_jsl_biobert
date: 2021-09-05
tags: [clinical, licensed, en, ner]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 3.2.0
spark_version: 3.0
supported: true
annotator: MedicalNerModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description


Pretrained named entity recognition deep learning model for clinical terminology. The SparkNLP deep learning model (MedicalNerModel) is inspired by a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM-CNN. This model is trained using BERT token embeddings `biobert_pubmed_base_cased`.


## Predicted Entities


`Strength`, `Pregnancy_Delivery_Puerperium`, `Female_Reproductive_Status`, `Fetus_NewBorn`, `Age`, `Alcohol`, `Treatment`, `Internal_organ_or_component`, `Vital_Signs_Header`, `Dosage`, `Employment`, `Gender`, `Disease_Syndrome_Disorder`, `Pregnancy`, `Symptom`, `Clinical_Dept`, `Medical_Device`, `Temperature`, `Hypertension`, `Cerebrovascular_Disease`, `Psychological_Condition`, `Respiration`, `Direction`, `Metastasis`, `Injury_or_Poisoning`, `Birth_Entity`, `Allergen`, `Labour_Delivery`, `Overweight`, `Family_History_Header`, `Section_Header`, `Diabetes`, `Hyperlipidemia`, `Death_Entity`, `Route`, `Duration`, `Admission_Discharge`, `Total_Cholesterol`, `Performance_Status`, `LDL`, `RelativeDate`, `Test_Result`, `Height`, `Procedure`, `Date`, `Cancer_Modifier`, `BMI`, `External_body_part_or_region`, `Kidney_Disease`, `Modifier`, `Oncology_Therapy`, `Drug_BrandName`, `Form`, `Substance`, `Social_History_Header`, `Obesity`, `Oncological`, `Sexually_Active_or_Sexual_Orientation`, `EKG_Findings`, `Oxygen_Therapy`, `Frequency`, `Relationship_Status`, `Communicable_Disease`, `Imaging_Technique`, `Vaccine`, `Pulse`, `Tumor_Finding`, `Heart_Disease`, `Time`, `ImagingFindings`, `HDL`, `O2_Saturation`, `Weight`, `Medical_History_Header`, `Blood_Pressure`, `Puerperium`, `Smoking`, `Substance_Quantity`, `RelativeTime`, `Test`, `Race_Ethnicity`, `Diet`, `Staging`, `Triglycerides`, `Drug_Ingredient`, `VS_Finding`


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_JSL/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_JSL.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_jsl_biobert_en_3.2.0_3.0_1630831908173.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


## How to use


<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = DocumentAssembler()\
		.setInputCol("text")\
		.setOutputCol("document")

sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models") \
		.setInputCols(["document"]) \
		.setOutputCol("sentence")

tokenizer = Tokenizer()\
		.setInputCols(["sentence"])\
		.setOutputCol("token")
	
embeddings = BertEmbeddings.pretrained("biobert_pubmed_base_cased")\
		.setInputCols(["sentence",  "token"]) \
		.setOutputCol("embeddings")
		
jsl_ner = MedicalNerModel.pretrained("ner_jsl_biobert", "en", "clinical/models") \
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

data = spark.createDataFrame([["The patient is a 21-day-old Caucasian male here for 2 days of congestion - mom has been suctioning yellow discharge from the patient's nares, plus she has noticed some mild problems with his breathing while feeding (but negative for any perioral cyanosis or retractions). One day ago, mom also noticed a tactile temperature and gave the patient Tylenol. Baby also has had some decreased p.o. intake. His normal breast-feeding is down from 20 minutes q.2h. to 5 to 10 minutes secondary to his respiratory congestion. He sleeps well, but has been more tired and has been fussy over the past 2 days. The parents noticed no improvement with albuterol treatments given in the ER. His urine output has also decreased; normally he has 8 to 10 wet and 5 dirty diapers per 24 hours, now he has down to 4 wet diapers per 24 hours. Mom denies any diarrhea. His bowel movements are yellow colored and soft in nature."]]).toDF("text")

result = jsl_ner_model.transform(data)
```
```scala
val documentAssembler = new DocumentAssembler()
		.setInputCol("text")
		.setOutputCol("document")

val sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models")
		.setInputCols("document") 
		.setOutputCol("sentence")

val tokenizer = new Tokenizer()
		.setInputCols("sentence")
		.setOutputCol("token")
	
val embeddings = BertEmbeddings.pretrained("biobert_pubmed_base_cased")
		.setInputCols(Array("sentence",  "token")) 
		.setOutputCol("embeddings")

val jsl_ner = MedicalNerModel.pretrained("ner_jsl_biobert", "en", "clinical/models")
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


{:.nlu-block}
```python
import nlu
nlu.load("en.med_ner.jsl.biobert").predict("""The patient is a 21-day-old Caucasian male here for 2 days of congestion - mom has been suctioning yellow discharge from the patient's nares, plus she has noticed some mild problems with his breathing while feeding (but negative for any perioral cyanosis or retractions). One day ago, mom also noticed a tactile temperature and gave the patient Tylenol. Baby also has had some decreased p.o. intake. His normal breast-feeding is down from 20 minutes q.2h. to 5 to 10 minutes secondary to his respiratory congestion. He sleeps well, but has been more tired and has been fussy over the past 2 days. The parents noticed no improvement with albuterol treatments given in the ER. His urine output has also decreased; normally he has 8 to 10 wet and 5 dirty diapers per 24 hours, now he has down to 4 wet diapers per 24 hours. Mom denies any diarrhea. His bowel movements are yellow colored and soft in nature.""")
```

</div>


## Results


```bash
|    | chunk                                     | entity                       |
|---:|:------------------------------------------|:-----------------------------|
|  0 | 21-day-old                                | Age                          |
|  1 | Caucasian                                 | Race_Ethnicity               |
|  2 | male                                      | Gender                       |
|  3 | for 2 days                                | Duration                     |
|  4 | congestion                                | Symptom                      |
|  5 | mom                                       | Gender                       |
|  6 | suctioning yellow discharge               | Symptom                      |
|  7 | nares                                     | External_body_part_or_region |
|  8 | she                                       | Gender                       |
|  9 | mild                                      | Modifier                     |
| 10 | problems with his breathing while feeding | Symptom                      |
| 11 | perioral cyanosis                         | Symptom                      |
| 12 | retractions                               | Symptom                      |
| 13 | One day ago                               | RelativeDate                 |
| 14 | mom                                       | Gender                       |
| 15 | tactile temperature                       | Symptom                      |
| 16 | Tylenol                                   | Drug_BrandName               |
| 17 | decreased p.o                             | Symptom                      |
| 18 | His                                       | Gender                       |
| 19 | from 20 minutes q.2h. to 5 to 10 minutes  | Frequency                    |
| 20 | his                                       | Gender                       |
| 21 | respiratory congestion                    | Symptom                      |
| 22 | He                                        | Gender                       |
| 23 | tired                                     | Symptom                      |
| 24 | fussy                                     | Symptom                      |
| 25 | over the past                             | RelativeDate                 |


```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|ner_jsl_biobert|
|Compatibility:|Healthcare NLP 3.2.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|


## Data Source


Trained on data gathered and manually annotated by John Snow Labs. https://www.johnsnowlabs.com/data/


## Benchmarking


```bash
label                                    tp     fp     fn     prec        rec         f1        
B-Oxygen_Therapy                         114    41     38     0.7354839   0.75        0.742671  
B-Cerebrovascular_Disease                42     16     19     0.7241379   0.6885246   0.7058824 
B-Triglycerides                          2      0      2      1           0.5         0.6666667 
I-Cerebrovascular_Disease                17     11     17     0.60714287  0.5         0.54838705
B-Medical_Device                         2568   334    400    0.88490695  0.8652291   0.8749574 
B-Labour_Delivery                        31     8      17     0.7948718   0.6458333   0.71264374
I-Vaccine                                12     3      4      0.8         0.75        0.7741936 
I-Obesity                                4      2      2      0.6666667   0.6666667   0.6666667 
B-RelativeTime                           126    71     94     0.6395939   0.57272726  0.60431653
B-Heart_Disease                          254    80     43     0.76047903  0.8552188   0.8050713 
B-Procedure                              2019   270    302    0.88204455  0.86988366  0.8759219 
I-RelativeTime                           183    93     44     0.6630435   0.8061674   0.72763425
B-Obesity                                46     5      5      0.9019608   0.9019608   0.9019608 
I-RelativeDate                           629    125    76     0.8342175   0.89219856  0.8622344 
B-O2_Saturation                          51     28     28     0.6455696   0.6455696   0.6455696 
B-Direction                              3016   219    360    0.93230295  0.8933649   0.9124187 
I-Alcohol                                3      2      3      0.6         0.5         0.54545456
I-Oxygen_Therapy                         91     67     28     0.5759494   0.7647059   0.6570397 
B-Dosage                                 277    82     86     0.7715877   0.7630854   0.767313  
B-Injury_or_Poisoning                    336    56     86     0.85714287  0.79620856  0.8255528 
B-Hypertension                           104    9      2      0.920354    0.9811321   0.9497717 
I-Test_Result                            1173   101    119    0.9207221   0.90789473  0.9142634 
B-Substance_Quantity                     4      8      0      0.33333334  1           0.5       
B-Alcohol                                68     9      6      0.8831169   0.9189189   0.90066224
B-Height                                 19     10     11     0.6551724   0.6333333   0.64406776
I-Substance                              10     2      6      0.8333333   0.625       0.71428573
B-RelativeDate                           416    91     58     0.82051283  0.87763715  0.84811425
B-Admission_Discharge                    245    12     7      0.9533074   0.9722222   0.9626719 
B-Date                                   316    17     14     0.9489489   0.95757574  0.9532428 
B-Kidney_Disease                         68     13     23     0.83950615  0.74725276  0.7906977 
I-Strength                               505    50     46     0.9099099   0.9165154   0.91320074
I-Injury_or_Poisoning                    255    73     132    0.777439    0.65891474  0.71328676
I-Drug_Ingredient                        279    102    38     0.7322835   0.8801262   0.799427  
I-Time                                   323    31     17     0.9124294   0.95        0.9308358 
B-Substance                              46     6      12     0.88461536  0.79310346  0.8363636 
B-Total_Cholesterol                      8      4      7      0.6666667   0.53333336  0.59259266
I-Vital_Signs_Header                     152    18     2      0.89411765  0.987013    0.9382716 
I-Internal_organ_or_component            2755   490    350    0.8489985   0.88727856  0.8677165 
B-Hyperlipidemia                         37     7      3      0.84090906  0.925       0.8809524 
I-Sexually_Active_or_Sexual_Orientation  5      0      0      1           1           1         
B-Sexually_Active_or_Sexual_Orientation  5      0      2      1           0.71428573  0.8333334 
I-Fetus_NewBorn                          44     60     28     0.42307693  0.6111111   0.5       
B-BMI                                    4      1      2      0.8         0.6666667   0.72727275
B-ImagingFindings                        71     40     83     0.6396396   0.46103895  0.53584903
B-Drug_Ingredient                        1636   235    222    0.8743987   0.8805167   0.877447  
B-Test_Result                            1369   180    188    0.883796    0.879255    0.8815196 
B-Section_Header                         2735   115    116    0.95964915  0.9593125   0.95948076
I-Treatment                              84     28     35     0.75        0.7058824   0.7272728 
B-Clinical_Dept                          721    101    89     0.87712896  0.8901235   0.8835784 
I-Kidney_Disease                         106    9      7      0.9217391   0.9380531   0.9298245 
I-Pulse                                  140    49     35     0.7407407   0.8         0.7692308 
B-Test                                   2267   375    390    0.8580621   0.8532179   0.85563314
B-Weight                                 70     16     16     0.81395346  0.81395346  0.81395346
I-Respiration                            61     5      28     0.92424244  0.6853933   0.78709674
I-EKG_Findings                           50     38     44     0.5681818   0.5319149   0.5494506 
I-Section_Header                         1998   108    65     0.94871795  0.9684925   0.95850325
I-VS_Finding                             36     31     29     0.53731346  0.5538462   0.5454546 
B-Strength                               541    51     54     0.9138514   0.9092437   0.9115417 
I-Social_History_Header                  43     3      5      0.9347826   0.8958333   0.9148936 
B-Vital_Signs_Header                     228    26     3      0.8976378   0.987013    0.94020617
B-Death_Entity                           30     5      4      0.85714287  0.88235295  0.86956525
B-Modifier                               2023   367    375    0.84644353  0.8436197   0.8450293 
B-Blood_Pressure                         110    23     32     0.8270677   0.7746479   0.8       
I-O2_Saturation                          93     56     29     0.62416106  0.76229507  0.6863469 
B-Frequency                              564    53     61     0.91410047  0.9024      0.9082126 
I-Triglycerides                          2      0      1      1           0.6666667   0.8       
I-Duration                               510    71     88     0.8777969   0.8528428   0.86513996
I-Diabetes                               35     2      5      0.9459459   0.875       0.9090909 
B-Race_Ethnicity                         67     2      4      0.9710145   0.943662    0.9571429 
I-Height                                 72     23     9      0.75789475  0.8888889   0.8181819 
B-Communicable_Disease                   12     5      8      0.7058824   0.6         0.6486487 
I-Family_History_Header                  57     3      1      0.95        0.98275864  0.9661017 
B-LDL                                    1      0      2      1           0.33333334  0.5       
B-Form                                   180    38     31     0.82568806  0.8530806   0.8391608 
I-Race_Ethnicity                         2      1      0      0.6666667   1           0.8       
B-Psychological_Condition                87     15     20     0.85294116  0.8130841   0.83253586
I-Drug_BrandName                         25     8      18     0.75757575  0.5813953   0.6578947 
I-Age                                    182    18     33     0.91        0.8465116   0.87710845
B-EKG_Findings                           41     19     24     0.68333334  0.63076925  0.65599996
B-Employment                             161    16     45     0.90960455  0.7815534   0.8407311 
I-Oncological                            338    32     62     0.91351354  0.845       0.8779221 
B-Time                                   335    42     19     0.88859415  0.9463277   0.91655266
B-Treatment                              98     43     63     0.69503546  0.6086956   0.6490066 
B-Temperature                            97     13     20     0.8818182   0.82905984  0.8546256 
I-Procedure                              2657   326    438    0.89071405  0.8584814   0.8743007 
B-Relationship_Status                    34     4      3      0.8947368   0.9189189   0.90666664
B-Pregnancy                              51     25     21     0.67105263  0.7083333   0.68918914
B-Fetus_NewBorn                          30     31     27     0.4918033   0.5263158   0.5084746 
I-Total_Cholesterol                      10     2      8      0.8333333   0.5555556   0.66666675
I-Route                                  205    16     21     0.9276018   0.90707964  0.91722596
I-Communicable_Disease                   6      4      2      0.6         0.75        0.6666667 
I-Medical_History_Header                 116    5      10     0.9586777   0.9206349   0.9392713 
B-Smoking                                85     4      3      0.9550562   0.96590906  0.960452  
I-Labour_Delivery                        30     5      22     0.85714287  0.5769231   0.6896552 
I-Death_Entity                           4      1      1      0.8         0.8         0.8000001 
B-Diabetes                               87     5      5      0.9456522   0.9456522   0.9456522 
B-HDL                                    1      1      0      0.5         1           0.6666667 
B-Drug_BrandName                         828    112    96     0.8808511   0.8961039   0.88841206
B-Gender                                 4420   61     62     0.98638695  0.9861669   0.98627687
B-Vaccine                                13     0      8      1           0.61904764  0.7647059 
I-Heart_Disease                          315    145    27     0.6847826   0.92105263  0.7855362 
I-Dosage                                 214    75     64     0.7404844   0.76978415  0.7548501 
B-Social_History_Header                  72     3      6      0.96        0.9230769   0.9411765 
B-External_body_part_or_region           1759   194    376    0.90066564  0.8238876   0.8605675 
I-Clinical_Dept                          531    43     52     0.9250871   0.9108062   0.9178911 
I-Test                                   1692   404    352    0.80725193  0.82778865  0.81739134
I-Frequency                              445    66     61     0.8708415   0.8794466   0.87512296
B-Age                                    492    28     39     0.9461538   0.92655367  0.9362512 
B-Pulse                                  86     31     30     0.73504275  0.7413793   0.7381974 
I-Symptom                                3072   1404   1050   0.6863271   0.7452693   0.71458477
I-Pregnancy                              43     25     26     0.63235295  0.6231884   0.6277372 
I-LDL                                    3      0      1      1           0.75        0.85714287
I-Diet                                   29     15     26     0.65909094  0.5272727   0.5858585 
I-Blood_Pressure                         171    52     35     0.76681614  0.8300971   0.79720277
I-ImagingFindings                        153    86     88     0.64016736  0.6348548   0.6375    
I-Date                                   184    10     9      0.9484536   0.9533679   0.9509044 
B-Route                                  726    77     80     0.9041096   0.90074444  0.9024239 
B-Duration                               212    29     50     0.87966806  0.8091603   0.84294236
B-Medical_History_Header                 89     8      5      0.91752577  0.9468085   0.9319371 
I-Metastasis                             5      0      1      1           0.8333333   0.90909094
B-Respiration                            49     10     18     0.8305085   0.73134327  0.77777773
I-External_body_part_or_region           431    49     133    0.8979167   0.7641844   0.82567054
I-BMI                                    13     2      3      0.8666667   0.8125      0.83870965
B-Internal_organ_or_component            4260   612    634    0.8743842   0.8704536   0.8724145 
I-Weight                                 177    42     16     0.8082192   0.91709846  0.8592233 
B-Disease_Syndrome_Disorder              2091   367    318    0.8506916   0.867995    0.85925627
B-Symptom                                4752   913    803    0.83883494  0.85544556  0.84705883
B-VS_Finding                             180    46     45     0.79646015  0.8         0.7982262 
I-Disease_Syndrome_Disorder              1592   331    309    0.8278731   0.83745396  0.832636  
I-Modifier                               148    96     128    0.60655737  0.5362319   0.56923074
I-Medical_Device                         1677   235    266    0.87709206  0.8630983   0.870039  
B-Oncological                            381    33     44     0.9202899   0.8964706   0.90822405
I-Temperature                            154    12     34     0.92771083  0.81914896  0.8700565 
I-Employment                             82     19     30     0.8118812   0.73214287  0.76995313
I-Psychological_Condition                25     2      7      0.9259259   0.78125     0.8474576 
B-Family_History_Header                  58     2      2      0.96666664  0.96666664  0.96666664
I-Direction                              189    29     49     0.8669725   0.7941176   0.8289474 
I-HDL                                    1      2      0      0.33333334  1           0.5       
Macro-average                            69137  11083  11027  0.7179756   0.7057431   0.7118068
Micro-average                            69137  11083  11027  0.8618424   0.8624444   0.86214334
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE1OTEzMTM1NjcsLTYyNzAwOTg3OSw5MT
EyOTkyMTksMTAwMTI5OTQyNiw5NTIyMzM5NTNdfQ==
-->
