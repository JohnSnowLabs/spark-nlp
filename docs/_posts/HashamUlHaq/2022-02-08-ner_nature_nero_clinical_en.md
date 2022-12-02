---
layout: model
title: Extract textual entities in biomedical texts
author: John Snow Labs
name: ner_nature_nero_clinical
date: 2022-02-08
tags: [ner, en, clinical, licensed]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 3.3.4
spark_version: 3.0
supported: true
annotator: MedicalNerModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description


This model is based on the NERO corpus, capable of extracting general entities. This model is trained to refute the claims made in https://www.nature.com/articles/s41540-021-00200-x regarding Spark NLP's performance and we hereby prove that we can get better than what is claimed. So, **this model is not meant to be used in production.**


## Predicted Entities


`Organismpart`, `Chromosome`, `Physicalphenomenon`, `Abstractconcept`, `Gene`, `Meas`, `Machineactivity`, `Warfarin`, `Gen`, `Aminoacidpeptide`, `Language`, `P`, `Quantityormeasurement`, `Disease`, `Process`, `Propernamedgeographicallocation`, `Duration`, `Medicalprocedureordevice`, `Citation`, `Geographicnotproper`, `Atom`, `Gp`, `Medicaldevice`, `Namedentity`, `Unpropernamedgeographicallocation`, `Persongroup`, `Unit`, `Bodypart`, `Unconjugated`, `Timepoint`, `Protein`, `Publishedsourceofinformation`, `Quantity`, `Dr`, `Organism`, `Nonproteinornucleicacidchemical`, `G`, `Researchactivity`, `Drug`, `Measurement`, `Cells`, `Journal`, `Relationshipphrase`, `Medicalprocedure`, `Geographiclocation`, `Groupofpeople`, `Person`, `Tissue`, `Mentalprocess`, `Facility`, `Chemical`, `Geneorproteingroup`, `Ion`, `Food`, `Aminoacid`, `N`, `Biologicalprocess`, `Cell`, `Researchactivty`, `Publicationorcitation`, `Molecularprocess`, `Experimentalfactor`, `Medicalfinding`, `Nucleicacid`, `Laboratoryexperimentalfactor`, `Relationship`, `Geographicallocation`, `Geneorprotein`, `Smallmolecule`, `Partofprotein`, `Thing`, `Quantityormeasure`, `Environmentalfactor`, `Intellectualproduct`, `R`, `Molecule`, `Time`, `Anatomicalpart`, `Cellcomponent`, `Nucleicacidsubstance`


{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_nature_nero_clinical_en_3.3.4_3.0_1644358495292.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")
         
sentence_detector = SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

embeddings_clinical = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("embeddings")

clinical_ner = MedicalNerModel.pretrained("ner_nature_nero_clinical", "en", "clinical/models") \
  .setInputCols(["sentence", "token", "embeddings"]) \
  .setOutputCol("ner")

ner_converter = NerConverter()\
 	  .setInputCols(["sentence", "token", "ner"])\
 	  .setOutputCol("ner_chunk")

nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings_clinical,  clinical_ner, ner_converter])

model = nlpPipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

results = model.transform(spark.createDataFrame([["The patient is a 21-day-old Caucasian male here for 2 days of congestion - mom has been suctioning yellow discharge from the patient's nares, plus she has noticed some mild problems with his breathing while feeding (but negative for any perioral cyanosis or retractions). One day ago, mom also noticed a tactile temperature and gave the patient Tylenol. Baby also has had some decreased p.o. intake. His normal breast-feeding is down from 20 minutes q.2h. to 5 to 10 minutes secondary to his respiratory congestion. He sleeps well, but has been more tired and has been fussy over the past 2 days. The parents noticed no improvement with albuterol treatments given in the ER. His urine output has also decreased; normally he has 8 to 10 wet and 5 dirty diapers per 24 hours, now he has down to 4 wet diapers per 24 hours. Mom denies any diarrhea. His bowel movements are yellow colored and soft in nature."]], ["text"]))
```
```scala
val document_assembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")
         
val sentence_detector = new SentenceDetector()
    .setInputCols("document")
    .setOutputCol("sentence")

val tokenizer = new Tokenizer()
    .setInputCols("sentence")
    .setOutputCol("token")

val embeddings_clinical = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
   .setInputCols(Array("sentence", "token"))
   .setOutputCol("embeddings")

val ner = MedicalNerModel.pretrained("ner_nature_nero_clinical", "en", "clinical/models") 
  .setInputCols(Array("sentence", "token", "embeddings"))
  .setOutputCol("ner")

val ner_converter = new NerConverter()
 	.setInputCols(Array("sentence", "token", "ner"))
 	.setOutputCol("ner_chunk")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings_clinical, ner, ner_converter))

val data = Seq("""The patient is a 21-day-old Caucasian male here for 2 days of congestion - mom has been suctioning yellow discharge from the patient's nares, plus she has noticed some mild problems with his breathing while feeding (but negative for any perioral cyanosis or retractions). One day ago, mom also noticed a tactile temperature and gave the patient Tylenol. Baby also has had some decreased p.o. intake. His normal breast-feeding is down from 20 minutes q.2h. to 5 to 10 minutes secondary to his respiratory congestion. He sleeps well, but has been more tired and has been fussy over the past 2 days. The parents noticed no improvement with albuterol treatments given in the ER. His urine output has also decreased; normally he has 8 to 10 wet and 5 dirty diapers per 24 hours, now he has down to 4 wet diapers per 24 hours. Mom denies any diarrhea. His bowel movements are yellow colored and soft in nature.""").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>


## Results


```bash
|    | chunk                                        | entity                |
|---:|:---------------------------------------------|:----------------------|
|  0 | perioral cyanosis                            | Medicalfinding        |
|  1 | One day                                      | Duration              |
|  2 | mom                                          | Namedentity           |
|  3 | tactile temperature                          | Quantityormeasurement |
|  4 | patient Tylenol                              | Chemical              |
|  5 | decreased p.o. intake                        | Medicalprocedure      |
|  6 | normal breast-feeding                        | Medicalfinding        |
|  7 | 20 minutes q.2h                              | Timepoint             |
|  8 | 5 to 10 minutes                              | Duration              |
|  9 | respiratory congestion                       | Medicalfinding        |
| 10 | past 2 days                                  | Duration              |
| 11 | parents                                      | Persongroup           |
| 12 | improvement                                  | Process               |
| 13 | albuterol treatments                         | Medicalprocedure      |
| 14 | ER                                           | Bodypart              |
| 15 | urine output                                 | Quantityormeasurement |
| 16 | 8 to 10 wet and 5 dirty diapers per 24 hours | Measurement           |
| 17 | 4 wet diapers per 24 hours                   | Measurement           |
| 18 | Mom                                          | Person                |
| 19 | diarrhea                                     | Medicalfinding        |
| 20 | bowel movements                              | Biologicalprocess     |
| 21 | soft in nature                               | Biologicalprocess     |


```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|ner_nature_nero_clinical|
|Compatibility:|Healthcare NLP 3.3.4+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|15.1 MB|


## References


This model is based on https://www.nature.com/articles/s41540-021-00200-x
and a response to: https://static-content.springer.com/esm/art%3A10.1038%2Fs41540-021-00200-x/MediaObjects/41540_2021_200_MOESM1_ESM.pdf


## Benchmarking


```bash
label	 tp	 fp	 fn	 prec	 rec	 f1
B-Atom	 11	 7	 48	 0.6111111	 0.18644068	 0.2857143
I-Laboratoryexperimentalfactor	 0	 3	 55	 0.0	 0.0	 0.0
B-Disease	 489	 232	 251	 0.6782247	 0.6608108	 0.6694045
B-Partofprotein	 77	 61	 67	 0.557971	 0.5347222	 0.54609925
B-Nonproteinornucleicacidchemical	 45	 58	 190	 0.4368932	 0.19148937	 0.2662722
I-Propernamedgeographicallocation	 60	 33	 31	 0.6451613	 0.6593407	 0.6521739
B-Bodypart	 648	 336	 343	 0.6585366	 0.65388495	 0.6562025
B-Protein	 832	 504	 440	 0.6227545	 0.6540881	 0.63803685
I-Unit	 0	 0	 4	 0.0	 0.0	 0.0
B-Chemical	 1390	 1066	 926	 0.5659609	 0.6001727	 0.5825649
B-Publicationorcitation	 2	 10	 9	 0.16666667	 0.18181819	 0.17391303
I-Smallmolecule	 34	 201	 353	 0.14468086	 0.087855294	 0.10932476
I-Abstractconcept	 0	 0	 4	 0.0	 0.0	 0.0
I-Nucleicacid	 449	 213	 249	 0.67824775	 0.6432665	 0.6602941
B-Drug	 306	 170	 213	 0.64285713	 0.5895954	 0.6150754
B-Thing	 0	 0	 19	 0.0	 0.0	 0.0
I-Citation	 6	 14	 17	 0.3	 0.26086956	 0.27906975
I-Aminoacid	 80	 43	 95	 0.6504065	 0.45714286	 0.53691274
B-Medicalprocedureordevice	 1	 0	 2	 1.0	 0.33333334	 0.5
I-Nucleicacidsubstance	 3	 3	 14	 0.5	 0.1764706	 0.26086956
I-Gp	 1244	 860	 427	 0.5912548	 0.7444644	 0.6590729
I-Geographicallocation	 0	 18	 17	 0.0	 0.0	 0.0
I-Molecule	 15	 114	 58	 0.11627907	 0.20547946	 0.14851485
B-R	 0	 0	 1	 0.0	 0.0	 0.0
I-Measurement	 1700	 893	 546	 0.6556113	 0.75690114	 0.7026245
I-Intellectualproduct	 197	 251	 311	 0.43973213	 0.38779527	 0.41213387
B-Anatomicalpart	 0	 22	 38	 0.0	 0.0	 0.0
B-Gp	 3597	 1426	 818	 0.71610594	 0.81472254	 0.7622378
B-Person	 105	 42	 80	 0.71428573	 0.5675676	 0.63253015
I-Aminoacidpeptide	 55	 38	 35	 0.5913978	 0.6111111	 0.6010929
B-Environmentalfactor	 24	 30	 40	 0.44444445	 0.375	 0.40677968
B-Cellcomponent	 188	 146	 191	 0.56287426	 0.49604222	 0.5273493
I-Groupofpeople	 1	 1	 10	 0.5	 0.09090909	 0.15384614
I-Chromosome	 39	 27	 12	 0.59090906	 0.7647059	 0.6666667
B-G	 0	 0	 1	 0.0	 0.0	 0.0
I-Publishedsourceofinformation	 122	 100	 131	 0.5495495	 0.48221344	 0.5136842
I-Disease	 710	 313	 239	 0.69403714	 0.74815595	 0.72008115
I-Time	 19	 32	 86	 0.37254903	 0.18095239	 0.24358974
I-Relationship	 41	 18	 33	 0.69491524	 0.5540541	 0.6165413
I-Nonproteinornucleicacidchemical	 32	 87	 257	 0.26890758	 0.11072665	 0.15686275
I-Molecularprocess	 1257	 1057	 589	 0.5432152	 0.68093175	 0.6043269
I-Persongroup	 587	 199	 233	 0.7468193	 0.71585363	 0.7310087
B-Laboratoryexperimentalfactor	 0	 2	 42	 0.0	 0.0	 0.0
I-Mentalprocess	 24	 30	 97	 0.44444445	 0.1983471	 0.2742857
B-Aminoacidpeptide	 33	 26	 44	 0.55932206	 0.42857143	 0.48529413
B-Food	 63	 29	 54	 0.6847826	 0.53846157	 0.6028708
B-Journal	 0	 0	 3	 0.0	 0.0	 0.0
I-Quantityormeasure	 0	 2	 4	 0.0	 0.0	 0.0
I-Cell	 1035	 212	 252	 0.829992	 0.8041958	 0.81689036
B-Tissue	 57	 41	 53	 0.5816327	 0.5181818	 0.5480769
I-Medicaldevice	 51	 58	 53	 0.4678899	 0.4903846	 0.47887325
B-Mentalprocess	 57	 49	 156	 0.5377358	 0.26760563	 0.35736677
I-Bodypart	 659	 406	 366	 0.61877936	 0.6429268	 0.630622
I-Researchactivity	 1073	 568	 410	 0.65386957	 0.7235334	 0.6869398
I-Atom	 11	 2	 51	 0.84615386	 0.17741935	 0.29333335
B-Namedentity	 173	 423	 504	 0.29026845	 0.25553915	 0.2717989
B-Quantityormeasure	 2	 2	 6	 0.5	 0.25	 0.33333334
B-Citation	 0	 2	 3	 0.0	 0.0	 0.0
I-Cellcomponent	 183	 166	 180	 0.5243553	 0.5041322	 0.51404494
B-Unit	 0	 0	 3	 0.0	 0.0	 0.0
I-Person	 41	 50	 33	 0.45054945	 0.5540541	 0.49696973
I-Quantityormeasurement	 202	 418	 557	 0.32580644	 0.26613966	 0.2929659
B-Organismpart	 23	 54	 41	 0.2987013	 0.359375	 0.32624114
B-Cell	 723	 206	 231	 0.7782562	 0.7578616	 0.76792353
I-Chemical	 1898	 1128	 832	 0.62723064	 0.6952381	 0.65948576
I-Medicalfinding	 1749	 1267	 1362	 0.5799072	 0.56219864	 0.5709156
B-Process	 1522	 1421	 1714	 0.51715934	 0.47033376	 0.49263635
I-Food	 75	 39	 60	 0.65789473	 0.5555556	 0.60240966
I-Duration	 344	 269	 169	 0.5611746	 0.6705653	 0.61101246
I-Experimentalfactor	 59	 173	 200	 0.25431034	 0.22779922	 0.24032587
I-Quantity	 742	 670	 750	 0.52549577	 0.49731904	 0.5110193
B-Physicalphenomenon	 1	 2	 11	 0.33333334	 0.083333336	 0.13333334
I-Medicalprocedureordevice	 3	 0	 3	 1.0	 0.5	 0.6666667
B-Aminoacid	 86	 41	 109	 0.6771653	 0.44102564	 0.5341615
B-Quantity	 613	 554	 645	 0.5252785	 0.4872814	 0.5055671
B-Cells	 0	 0	 2	 0.0	 0.0	 0.0
I-Gene	 134	 44	 95	 0.752809	 0.58515286	 0.65847665
B-Medicalfinding	 1913	 1389	 1296	 0.5793458	 0.59613585	 0.5876209
I-Tissue	 70	 58	 42	 0.546875	 0.625	 0.5833333
B-Molecule	 20	 48	 62	 0.29411766	 0.24390244	 0.26666665
I-Organism	 959	 373	 282	 0.71997	 0.7727639	 0.74543333
I-Medicalprocedure	 775	 475	 370	 0.62	 0.6768559	 0.64718163
B-Unpropernamedgeographicallocation	 7	 5	 25	 0.5833333	 0.21875	 0.3181818
I-Timepoint	 3	 9	 32	 0.25	 0.08571429	 0.12765957
I-Organismpart	 15	 47	 27	 0.24193548	 0.35714287	 0.28846154
I-Biologicalprocess	 798	 787	 835	 0.50347	 0.48867115	 0.49596024
B-Time	 47	 49	 97	 0.48958334	 0.3263889	 0.39166668
B-Experimentalfactor	 40	 100	 136	 0.2857143	 0.22727273	 0.2531646
B-Nucleicacid	 364	 234	 337	 0.6086956	 0.5192582	 0.5604311
B-Propernamedgeographicallocation	 112	 38	 22	 0.74666667	 0.8358209	 0.7887324
B-Publishedsourceofinformation	 252	 95	 125	 0.7262248	 0.66843504	 0.69613266
I-Unpropernamedgeographicallocation	 5	 4	 21	 0.5555556	 0.1923077	 0.2857143
I-Protein	 1114	 717	 449	 0.6084107	 0.7127319	 0.65645254
B-Molecularprocess	 907	 696	 609	 0.5658141	 0.59828496	 0.5815967
B-Quantityormeasurement	 171	 293	 438	 0.36853448	 0.28078818	 0.31873253
B-Intellectualproduct	 159	 175	 215	 0.4760479	 0.42513368	 0.44915253
B-Persongroup	 783	 245	 223	 0.76167315	 0.77833	 0.7699115
I-Cells	 0	 0	 2	 0.0	 0.0	 0.0
B-Researchactivity	 869	 448	 373	 0.65983295	 0.69967794	 0.67917156
I-Environmentalfactor	 20	 18	 35	 0.5263158	 0.36363637	 0.43010756
B-Gene	 64	 37	 105	 0.63366336	 0.37869823	 0.47407413
B-Groupofpeople	 3	 2	 10	 0.6	 0.23076923	 0.33333334
B-Geneorproteingroup	 241	 191	 203	 0.5578704	 0.5427928	 0.5502283
B-Facility	 119	 134	 180	 0.47035572	 0.3979933	 0.43115944
B-Timepoint	 10	 15	 30	 0.4	 0.25	 0.30769232
B-Organism	 869	 290	 297	 0.7497843	 0.745283	 0.7475268
B-Duration	 274	 186	 137	 0.59565216	 0.6666667	 0.6291619
I-Facility	 182	 212	 226	 0.46192893	 0.44607842	 0.45386532
I-Process	 589	 880	 1114	 0.40095302	 0.34586024	 0.3713745
B-Language	 0	 0	 2	 0.0	 0.0	 0.0
B-Medicaldevice	 29	 44	 56	 0.39726028	 0.34117648	 0.36708862
B-Ion	 54	 23	 37	 0.7012987	 0.5934066	 0.64285713
I-Partofprotein	 126	 131	 153	 0.49027237	 0.4516129	 0.47014928
B-Gen	 0	 0	 1	 0.0	 0.0	 0.0
B-Geneorprotein	 0	 1	 64	 0.0	 0.0	 0.0
I-Thing	 0	 0	 16	 0.0	 0.0	 0.0
I-Gen	 0	 0	 3	 0.0	 0.0	 0.0
I-Geneorproteingroup	 398	 345	 286	 0.5356662	 0.58187133	 0.5578136
B-Abstractconcept	 0	 0	 7	 0.0	 0.0	 0.0
B-Chromosome	 37	 24	 24	 0.60655737	 0.60655737	 0.60655737
B-Relationship	 133	 64	 59	 0.6751269	 0.6927083	 0.6838046
B-Smallmolecule	 19	 62	 229	 0.2345679	 0.076612905	 0.115501516
I-Physicalphenomenon	 0	 2	 7	 0.0	 0.0	 0.0
I-Ion	 102	 41	 126	 0.7132867	 0.4473684	 0.5498652
I-Drug	 173	 133	 169	 0.5653595	 0.50584793	 0.5339506
I-Anatomicalpart	 0	 63	 48	 0.0	 0.0	 0.0
B-Measurement	 639	 451	 298	 0.5862385	 0.68196374	 0.6304884
I-Publicationorcitation	 10	 46	 52	 0.17857143	 0.16129032	 0.16949153
B-Geographicallocation	 4	 16	 20	 0.2	 0.16666667	 0.18181819
I-Journal	 0	 0	 11	 0.0	 0.0	 0.0
B-Relationshipphrase	 0	 0	 1	 0.0	 0.0	 0.0
B-Nucleicacidsubstance	 2	 2	 16	 0.5	 0.11111111	 0.18181819
B-Biologicalprocess	 779	 703	 831	 0.525641	 0.48385093	 0.503881
I-Geneorprotein	 0	 2	 33	 0.0	 0.0	 0.0
B-Medicalprocedure	 820	 450	 346	 0.6456693	 0.703259	 0.6732348
I-Namedentity	 140	 447	 381	 0.23850085	 0.268714	 0.25270757
Macro-average	 41221  28282  28209  0.4370514 0.3767836 0.40468597
Micro-average	 41221  28282  28209  0.5930823 0.5937059 0.5933939


```
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTIwMjAwNTE1MDgsLTk2MDkyMzc1NywtMj
A4OTAwMzIzMl19
-->