---
layout: model
title: Detect Clinical Entities (ner_jsl_enriched)
author: John Snow Labs
name: ner_jsl_enriched
date: 2021-10-22
tags: [ner, licensed, clinical, en]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 3.3.0
spark_version: 3.0
supported: true
recommended: true
annotator: MedicalNerModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description


Pretrained named entity recognition deep learning model for clinical terminology. This model is capable of predicting up to `87` different entities and is based on `ner_jsl`.


## Predicted Entities


`Social_History_Header`, `Oncology_Therapy`, `Blood_Pressure`, `Respiration`, `Performance_Status`, `Family_History_Header`, `Dosage`, `Clinical_Dept`, `Diet`, `Procedure`, `HDL`, `Weight`, `Admission_Discharge`, `LDL`, `Kidney_Disease`, `Oncological`, `Route`, `Imaging_Technique`, `Puerperium`, `Overweight`, `Temperature`, `Diabetes`, `Vaccine`, `Age`, `Test_Result`, `Employment`, `Time`, `Obesity`, `EKG_Findings`, `Pregnancy`, `Communicable_Disease`, `BMI`, `Strength`, `Tumor_Finding`, `Section_Header`, `RelativeDate`, `ImagingFindings`, `Death_Entity`, `Date`, `Cerebrovascular_Disease`, `Treatment`, `Labour_Delivery`, `Pregnancy_Delivery_Puerperium`, `Direction`, `Internal_organ_or_component`, `Psychological_Condition`, `Form`, `Medical_Device`, `Test`, `Symptom`, `Disease_Syndrome_Disorder`, `Staging`, `Birth_Entity`, `Hyperlipidemia`, `O2_Saturation`, `Frequency`, `External_body_part_or_region`, `Drug_Ingredient`, `Vital_Signs_Header`, `Substance_Quantity`, `Race_Ethnicity`, `VS_Finding`, `Injury_or_Poisoning`, `Medical_History_Header`, `Alcohol`, `Triglycerides`, `Total_Cholesterol`, `Sexually_Active_or_Sexual_Orientation`, `Female_Reproductive_Status`, `Relationship_Status`, `Drug_BrandName`, `RelativeTime`, `Duration`, `Hypertension`, `Metastasis`, `Gender`, `Oxygen_Therapy`, `Pulse`, `Heart_Disease`, `Modifier`, `Allergen`, `Smoking`, `Substance`, `Cancer_Modifier`, `Fetus_NewBorn`, `Height`


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_JSL/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_JSL.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_jsl_enriched_en_3.3.0_3.0_1634865045033.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_jsl_enriched_en_3.3.0_3.0_1634865045033.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}


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
	
embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
		.setInputCols(["sentence", "token"])\
		.setOutputCol("embeddings")

jsl_ner = MedicalNerModel.pretrained("ner_jsl_enriched", "en", "clinical/models") \
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

val sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare", "en", "clinical/models")
		.setInputCols("document") 
		.setOutputCol("sentence")

val tokenizer = new Tokenizer()
		.setInputCols("sentence")
		.setOutputCol("token")
	
val embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
		.setInputCols(Array("sentence", "token"))
	    .setOutputCol("embeddings")

val jsl_ner = MedicalNerModel.pretrained("ner_jsl_enriched", "en", "clinical/models")
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
nlu.load("en.med_ner.jsl.enriched").predict("""The patient is a 21-day-old Caucasian male here for 2 days of congestion - mom has been suctioning yellow discharge from the patient's nares, plus she has noticed some mild problems with his breathing while feeding (but negative for any perioral cyanosis or retractions). One day ago, mom also noticed a tactile temperature and gave the patient Tylenol. Baby also has had some decreased p.o. intake. His normal breast-feeding is down from 20 minutes q.2h. to 5 to 10 minutes secondary to his respiratory congestion. He sleeps well, but has been more tired and has been fussy over the past 2 days. The parents noticed no improvement with albuterol treatments given in the ER. His urine output has also decreased; normally he has 8 to 10 wet and 5 dirty diapers per 24 hours, now he has down to 4 wet diapers per 24 hours. Mom denies any diarrhea. His bowel movements are yellow colored and soft in nature.""")
```

</div>


## Results


```bash
|    | chunk                                     |   begin |   end | entity                       |
|---:|:------------------------------------------|--------:|------:|:-----------------------------|
|  0 | 21-day-old                                |      17 |    26 | Age                          |
|  1 | Caucasian                                 |      28 |    36 | Race_Ethnicity               |
|  2 | male                                      |      38 |    41 | Gender                       |
|  3 | 2 days                                    |      52 |    57 | Duration                     |
|  4 | congestion                                |      62 |    71 | Symptom                      |
|  5 | mom                                       |      75 |    77 | Gender                       |
|  6 | suctioning yellow discharge               |      88 |   114 | Symptom                      |
|  7 | nares                                     |     135 |   139 | External_body_part_or_region |
|  8 | she                                       |     147 |   149 | Gender                       |
|  9 | mild                                      |     168 |   171 | Modifier                     |
| 10 | problems with his breathing while feeding |     173 |   213 | Symptom                      |
| 11 | perioral cyanosis                         |     237 |   253 | Symptom                      |
| 12 | retractions                               |     258 |   268 | Symptom                      |
| 13 | One day ago                               |     272 |   282 | RelativeDate                 |
| 14 | mom                                       |     285 |   287 | Gender                       |
| 15 | tactile temperature                       |     304 |   322 | Symptom                      |
| 16 | Tylenol                                   |     345 |   351 | Drug_BrandName               |
| 17 | Baby                                      |     354 |   357 | Age                          |
| 18 | decreased p.o. intake                     |     377 |   397 | Symptom                      |
| 19 | His                                       |     400 |   402 | Gender                       |
| 20 | q.2h                                      |     450 |   453 | Frequency                    |
| 21 | 5 to 10 minutes                           |     459 |   473 | Duration                     |
| 22 | his                                       |     488 |   490 | Gender                       |
| 23 | respiratory congestion                    |     492 |   513 | Symptom                      |
| 24 | He                                        |     516 |   517 | Gender                       |
| 25 | tired                                     |     550 |   554 | Symptom                      |
| 26 | fussy                                     |     569 |   573 | Symptom                      |
| 27 | over the past 2 days                      |     575 |   594 | RelativeDate                 |
| 28 | albuterol                                 |     637 |   645 | Drug_Ingredient              |
| 29 | ER                                        |     671 |   672 | Clinical_Dept                |
| 30 | His                                       |     675 |   677 | Gender                       |
| 31 | urine output has also decreased           |     679 |   709 | Symptom                      |
| 32 | he                                        |     721 |   722 | Gender                       |
| 33 | per 24 hours                              |     760 |   771 | Frequency                    |
| 34 | he                                        |     778 |   779 | Gender                       |
| 35 | per 24 hours                              |     807 |   818 | Frequency                    |
| 36 | Mom                                       |     821 |   823 | Gender                       |
| 37 | diarrhea                                  |     836 |   843 | Symptom                      |
| 38 | His                                       |     846 |   848 | Gender                       |
| 39 | bowel                                     |     850 |   854 | Internal_organ_or_component  |
```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|ner_jsl_enriched|
|Compatibility:|Healthcare NLP 3.3.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|


## Data Source


Trained on data sampled from MTSamples and Clinicaltrials.gov, and annotated in-house.


## Benchmarking


```bash
label	 tp	 fp	 fn	 prec	 rec	 f1
B-Oxygen_Therapy	 139	 44	 44	 0.75956285	 0.75956285	 0.75956285
B-Oncology_Therapy	 2	 0	 4	 1.0	 0.33333334	 0.5
B-Cerebrovascular_Disease	 49	 13	 23	 0.7903226	 0.6805556	 0.7313434
B-Triglycerides	 3	 0	 1	 1.0	 0.75	 0.85714287
I-Cerebrovascular_Disease	 18	 11	 22	 0.62068963	 0.45	 0.52173907
B-Medical_Device	 2723	 350	 299	 0.88610476	 0.9010589	 0.8935192
B-Labour_Delivery	 38	 6	 27	 0.8636364	 0.5846154	 0.6972478
I-Vaccine	 27	 2	 8	 0.9310345	 0.7714286	 0.84374994
I-Obesity	 7	 0	 0	 1.0	 1.0	 1.0
I-Smoking	 2	 3	 4	 0.4	 0.33333334	 0.36363637
B-RelativeTime	 141	 69	 70	 0.67142856	 0.66824645	 0.6698337
I-Staging	 0	 0	 1	 0.0	 0.0	 0.0
B-Imaging_Technique	 23	 7	 30	 0.76666665	 0.43396226	 0.55421686
B-Heart_Disease	 264	 54	 51	 0.8301887	 0.83809525	 0.8341232
B-Procedure	 2091	 206	 277	 0.9103178	 0.8830236	 0.89646304
I-RelativeTime	 177	 56	 65	 0.75965667	 0.73140496	 0.74526316
I-Substance_Quantity	 0	 12	 1	 0.0	 0.0	 0.0
B-Obesity	 53	 0	 4	 1.0	 0.9298246	 0.9636364
I-RelativeDate	 702	 94	 97	 0.88190955	 0.8785983	 0.8802508
B-O2_Saturation	 55	 27	 22	 0.6707317	 0.71428573	 0.6918239
B-Direction	 3138	 213	 260	 0.9364369	 0.9234844	 0.92991555
I-Alcohol	 2	 0	 4	 1.0	 0.33333334	 0.5
I-Oxygen_Therapy	 104	 60	 57	 0.63414633	 0.6459627	 0.64
B-Diet	 34	 5	 39	 0.8717949	 0.46575344	 0.6071429
B-Dosage	 267	 59	 115	 0.8190184	 0.69895285	 0.7542373
B-Injury_or_Poisoning	 353	 67	 67	 0.8404762	 0.8404762	 0.8404762
B-Hypertension	 98	 3	 9	 0.97029704	 0.91588783	 0.94230765
I-Test_Result	 1093	 58	 145	 0.94960904	 0.8828756	 0.91502714
B-Female_Reproductive_Status	 0	 0	 1	 0.0	 0.0	 0.0
B-Substance_Quantity	 0	 4	 1	 0.0	 0.0	 0.0
B-Alcohol	 72	 6	 15	 0.9230769	 0.82758623	 0.8727273
B-Height	 14	 7	 8	 0.6666667	 0.6363636	 0.65116286
I-Substance	 19	 2	 4	 0.9047619	 0.82608694	 0.86363643
B-RelativeDate	 470	 65	 79	 0.8785047	 0.856102	 0.86715865
B-Admission_Discharge	 242	 8	 3	 0.968	 0.9877551	 0.9777778
B-Date	 424	 25	 18	 0.94432074	 0.959276	 0.9517396
B-Kidney_Disease	 71	 12	 12	 0.85542166	 0.85542166	 0.85542166
I-Admission_Discharge	 0	 0	 1	 0.0	 0.0	 0.0
I-Strength	 506	 82	 38	 0.8605442	 0.93014705	 0.89399296
B-Allergen	 0	 3	 10	 0.0	 0.0	 0.0
I-Injury_or_Poisoning	 315	 83	 93	 0.7914573	 0.77205884	 0.7816377
I-Drug_Ingredient	 300	 88	 46	 0.77319586	 0.867052	 0.8174387
I-Time	 298	 31	 14	 0.90577507	 0.9551282	 0.9297972
B-Substance	 54	 7	 9	 0.8852459	 0.85714287	 0.87096775
B-Total_Cholesterol	 12	 2	 3	 0.85714287	 0.8	 0.82758623
I-Vital_Signs_Header	 138	 8	 6	 0.94520545	 0.9583333	 0.9517241
I-Internal_organ_or_component	 2826	 302	 304	 0.9034527	 0.9028754	 0.903164
B-Hyperlipidemia	 27	 1	 1	 0.96428573	 0.96428573	 0.9642857
I-Sexually_Active_or_Sexual_Orientation	 4	 2	 1	 0.6666667	 0.8	 0.72727275
B-Sexually_Active_or_Sexual_Orientation	 4	 3	 2	 0.5714286	 0.6666667	 0.61538464
I-Fetus_NewBorn	 27	 18	 19	 0.6	 0.5869565	 0.5934066
B-BMI	 5	 0	 4	 1.0	 0.5555556	 0.71428573
B-ImagingFindings	 63	 36	 64	 0.6363636	 0.496063	 0.5575221
B-Drug_Ingredient	 1905	 202	 183	 0.9041291	 0.9123563	 0.90822405
B-Test_Result	 1327	 131	 184	 0.9101509	 0.87822634	 0.8939037
B-Section_Header	 2763	 120	 106	 0.9583767	 0.96305335	 0.96070933
I-Treatment	 103	 40	 39	 0.7202797	 0.7253521	 0.72280705
B-Clinical_Dept	 744	 62	 99	 0.9230769	 0.8825623	 0.902365
I-Kidney_Disease	 109	 12	 1	 0.90082645	 0.9909091	 0.94372296
I-Pulse	 156	 35	 27	 0.8167539	 0.852459	 0.8342246
B-Test	 2312	 293	 418	 0.887524	 0.84688646	 0.8667291
B-Weight	 64	 10	 14	 0.8648649	 0.82051283	 0.8421053
I-Respiration	 81	 47	 11	 0.6328125	 0.8804348	 0.73636365
I-EKG_Findings	 73	 15	 70	 0.82954544	 0.5104895	 0.63203466
I-Section_Header	 1999	 73	 97	 0.96476835	 0.95372134	 0.9592131
I-VS_Finding	 32	 17	 28	 0.6530612	 0.53333336	 0.58715594
B-Strength	 513	 60	 44	 0.89528793	 0.92100537	 0.9079646
I-Cancer_Modifier	 5	 0	 0	 1.0	 1.0	 1.0
I-Social_History_Header	 39	 6	 0	 0.8666667	 1.0	 0.92857146
B-Vital_Signs_Header	 216	 14	 6	 0.9391304	 0.972973	 0.95575225
B-Death_Entity	 41	 11	 3	 0.78846157	 0.9318182	 0.8541667
B-Modifier	 2050	 335	 307	 0.8595388	 0.86974967	 0.86461407
B-Blood_Pressure	 108	 17	 27	 0.864	 0.8	 0.83076924
I-O2_Saturation	 99	 23	 36	 0.8114754	 0.73333335	 0.77042806
B-Frequency	 519	 53	 63	 0.9073427	 0.8917526	 0.8994801
I-Triglycerides	 3	 0	 5	 1.0	 0.375	 0.54545456
I-Female_Reproductive_Status	 0	 0	 3	 0.0	 0.0	 0.0
I-Duration	 529	 71	 112	 0.88166666	 0.82527304	 0.8525383
I-Diabetes	 41	 8	 1	 0.8367347	 0.97619045	 0.90109897
B-Race_Ethnicity	 77	 0	 4	 1.0	 0.9506173	 0.9746836
I-Gender	 0	 0	 2	 0.0	 0.0	 0.0
I-Height	 40	 1	 18	 0.9756098	 0.6896552	 0.8080808
B-Communicable_Disease	 11	 2	 8	 0.84615386	 0.57894737	 0.68749994
I-Family_History_Header	 35	 0	 1	 1.0	 0.9722222	 0.9859155
B-LDL	 3	 1	 0	 0.75	 1.0	 0.85714287
B-Form	 169	 41	 40	 0.8047619	 0.80861247	 0.8066826
I-Race_Ethnicity	 2	 0	 2	 1.0	 0.5	 0.6666667
B-Psychological_Condition	 114	 12	 19	 0.9047619	 0.85714287	 0.88030887
I-Drug_BrandName	 14	 12	 12	 0.53846157	 0.53846157	 0.53846157
I-Hypertension	 2	 2	 10	 0.5	 0.16666667	 0.25
I-Age	 196	 43	 7	 0.8200837	 0.9655172	 0.88687783
B-EKG_Findings	 38	 18	 35	 0.6785714	 0.5205479	 0.58914727
B-Employment	 193	 31	 41	 0.86160713	 0.8247863	 0.8427947
I-Oncological	 333	 38	 23	 0.8975741	 0.9353933	 0.9160936
B-Time	 320	 34	 23	 0.9039548	 0.9329446	 0.91822094
B-Treatment	 129	 36	 61	 0.7818182	 0.6789474	 0.7267606
B-Temperature	 104	 15	 19	 0.8739496	 0.8455285	 0.85950416
B-Tumor_Finding	 1	 2	 10	 0.33333334	 0.09090909	 0.14285715
I-Procedure	 2667	 348	 335	 0.8845771	 0.8884077	 0.8864883
B-Relationship_Status	 37	 3	 3	 0.925	 0.925	 0.925
B-Pregnancy	 77	 17	 15	 0.81914896	 0.8369565	 0.827957
B-Fetus_NewBorn	 18	 7	 18	 0.72	 0.5	 0.59016395
I-Total_Cholesterol	 14	 1	 5	 0.93333334	 0.7368421	 0.8235294
I-Route	 193	 17	 13	 0.9190476	 0.9368932	 0.92788464
B-Birth_Entity	 1	 7	 1	 0.125	 0.5	 0.2
I-Communicable_Disease	 5	 1	 2	 0.8333333	 0.71428573	 0.7692307
I-Medical_History_Header	 119	 0	 3	 1.0	 0.97540987	 0.98755187
I-Imaging_Technique	 10	 1	 15	 0.90909094	 0.4	 0.5555555
B-Smoking	 96	 5	 5	 0.95049506	 0.95049506	 0.95049506
I-Labour_Delivery	 29	 20	 9	 0.59183675	 0.7631579	 0.6666667
I-Death_Entity	 3	 1	 0	 0.75	 1.0	 0.85714287
B-Diabetes	 77	 3	 3	 0.9625	 0.9625	 0.9625
B-HDL	 2	 0	 1	 1.0	 0.6666667	 0.8
B-Drug_BrandName	 792	 67	 61	 0.9220023	 0.9284877	 0.92523366
B-Gender	 4498	 58	 63	 0.9872695	 0.9861872	 0.9867281
B-Metastasis	 5	 2	 8	 0.71428573	 0.3846154	 0.5
I-Relationship_Status	 0	 0	 4	 0.0	 0.0	 0.0
B-Cancer_Modifier	 4	 0	 1	 1.0	 0.8	 0.88888896
B-Vaccine	 39	 6	 7	 0.8666667	 0.84782606	 0.8571428
I-Heart_Disease	 317	 47	 47	 0.8708791	 0.8708791	 0.8708791
I-Dosage	 216	 47	 126	 0.82129276	 0.6315789	 0.7140496
B-Staging	 0	 0	 2	 0.0	 0.0	 0.0
B-Social_History_Header	 65	 8	 3	 0.89041096	 0.9558824	 0.92198586
B-External_body_part_or_region	 1792	 195	 229	 0.9018621	 0.8866898	 0.8942116
I-Clinical_Dept	 559	 23	 47	 0.9604811	 0.92244226	 0.9410775
I-Tumor_Finding	 0	 13	 11	 0.0	 0.0	 0.0
I-Test	 1919	 311	 305	 0.8605381	 0.8628597	 0.8616973
I-Frequency	 447	 53	 68	 0.894	 0.86796117	 0.88078815
B-Age	 461	 50	 22	 0.90215266	 0.9544513	 0.9275654
B-Pulse	 96	 14	 18	 0.8727273	 0.84210527	 0.85714287
I-Symptom	 3408	 1152	 1091	 0.7473684	 0.75750166	 0.7524009
I-Form	 1	 5	 2	 0.16666667	 0.33333334	 0.22222224
I-Pregnancy	 66	 8	 36	 0.8918919	 0.64705884	 0.75000006
I-LDL	 5	 2	 6	 0.71428573	 0.45454547	 0.5555556
I-Diet	 40	 7	 25	 0.85106385	 0.61538464	 0.7142857
I-Blood_Pressure	 165	 35	 36	 0.825	 0.8208955	 0.8229426
I-ImagingFindings	 118	 60	 72	 0.66292137	 0.6210526	 0.6413044
I-Date	 195	 11	 10	 0.9466019	 0.9512195	 0.9489051
I-Hyperlipidemia	 1	 1	 0	 0.5	 1.0	 0.6666667
B-Route	 755	 69	 83	 0.91626215	 0.90095466	 0.90854394
B-Duration	 219	 30	 72	 0.8795181	 0.7525773	 0.81111115
B-Medical_History_Header	 84	 6	 3	 0.93333334	 0.9655172	 0.9491525
I-Metastasis	 3	 4	 4	 0.42857143	 0.42857143	 0.42857143
I-Allergen	 0	 1	 3	 0.0	 0.0	 0.0
B-Respiration	 53	 19	 15	 0.7361111	 0.7794118	 0.75714284
I-External_body_part_or_region	 429	 73	 62	 0.85458165	 0.8737271	 0.86404836
I-BMI	 12	 3	 3	 0.8	 0.8	 0.8000001
B-Internal_organ_or_component	 4361	 475	 509	 0.90177834	 0.89548254	 0.8986194
I-Weight	 146	 14	 21	 0.9125	 0.8742515	 0.8929664
B-Disease_Syndrome_Disorder	 2222	 283	 350	 0.88702595	 0.86391914	 0.8753201
B-Symptom	 4910	 711	 744	 0.87351006	 0.8684117	 0.87095344
B-VS_Finding	 207	 28	 44	 0.8808511	 0.8247012	 0.8518518
I-Disease_Syndrome_Disorder	 1659	 201	 383	 0.89193547	 0.8124388	 0.85033315
I-Modifier	 162	 67	 138	 0.70742357	 0.54	 0.61247635
I-Medical_Device	 1786	 245	 176	 0.8793698	 0.9102956	 0.89456546
B-Oncological	 354	 44	 29	 0.8894472	 0.92428195	 0.90653
I-Temperature	 172	 16	 24	 0.9148936	 0.877551	 0.8958333
I-Employment	 108	 18	 32	 0.85714287	 0.7714286	 0.81203
I-Psychological_Condition	 40	 9	 7	 0.81632656	 0.85106385	 0.8333334
B-Family_History_Header	 47	 0	 4	 1.0	 0.92156863	 0.9591837
I-Direction	 186	 42	 23	 0.81578946	 0.8899522	 0.8512586
I-HDL	 3	 0	 2	 1.0	 0.6	 0.75
Macro-average 71581  9121  10180  0.7729799  0.721845   0.7465378
Micro-average 71581  9121  10180  0.8869793  0.8754908  0.88119763
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTg0MzI4NzUyNCwxMzMwMDkyMTAwLDE2NT
UyMjQyNjRdfQ==
-->
