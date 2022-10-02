---
layout: docs
header: true
seotitle: Spark NLP for Healthcare | John Snow Labs
title: Spark NLP for Healthcare Release Notes 3.2.0
permalink: /docs/en/spark_nlp_healthcare_versions/release_notes_3_2_0
key: docs-licensed-release-notes
modify_date: 2021-07-14
show_nav: true
sidebar:
    nav: sparknlp-healthcare
---

## 3.2.0
We are glad to announce that Spark NLP Healthcare 3.2.0 has been released!.

#### Highlights

+ New Sentence Boundary Detection Model for Healthcare text
+ New Assertion Status Models
+ New Sentence Entity Resolver Model
+ Finetuning Sentence Entity Resolvers with Your Data
+ New Clinical NER Models
+ New CMS-HCC risk-adjustment score calculation module
+ New Embedding generation module for entity resolution


##### New Sentence Boundary Detection Model for Healthcare text

We are releasing an updated Sentence Boundary detection model to identify complex sentences containing multiple measurements, and punctuations. This model is trained on an in-house dataset.

*Example*:

*Python:*
```bash
...
documenter = DocumentAssembler()\
  .setInputCol("text")\
  .setOutputCol("document")

sentencerDL = SentenceDetectorDLModel
  .pretrained("sentence_detector_dl_healthcare","en","clinical/models")
  .setInputCols(["document"])
  .setOutputCol("sentences")

text = """He was given boluses of MS04 with some effect.he has since been placed on a PCA . He takes 80 mg. of ativan at home ativan for anxiety,
with 20 meq kcl po, 30 mmol K-phos iv and 2 gms mag so4 iv.
Size: Prostate gland measures 10x1.1x 4.9 cm (LS x AP x TS). Estimated volume is
51.9 ml. and is mildly enlarged in size.Normal delineation pattern of the prostate gland is preserved.
"""

sd_model = LightPipeline(PipelineModel(stages=[documenter, sentencerDL]))

result = sd_model.fullAnnotate(text)
```

*Results*:

```
| s.no | sentences                                                      |
|-----:|:---------------------------------------------------------------|
|    0 | He was given boluses of MS04 with some effect.                 |
|    1 | he has since been placed on a PCA .                            |
|    2 | He takes 80 mg. of ativan at home ativan for anxiety,          |
|      | with 20 meq kcl po, 30 mmol K-phos iv and 2 gms mag so4 iv.    |
|    3 | Size: Prostate gland measures 10x1.1x 4.9 cm (LS x AP x TS).   |
|    4 | Estimated volume is                                            |
|      | 51.9 ml. and is mildly enlarged in size.                       |
|    5 | Normal delineation pattern of the prostate gland is preserved. |

```

##### New Assertion Status Models

We are releasing two new Assertion Status Models based on the BiLSTM architecture. Apart from what we released in other assertion models, an in-house annotations on a curated dataset (6K clinical notes) is used to augment the base assertion dataset (2010 i2b2/VA).

+ `assertion_jsl`: This model can classify the assertions made on given medical concepts as being `Present`, `Absent`, `Possible`, `Planned`, `Someoneelse`, `Past`, `Family`, `None`, `Hypotetical`.

+ `assertion_jsl_large`: This model can classify the assertions made on given medical concepts as being `present`, `absent`, `possible`, `planned`, `someoneelse`, `past`.

*assertion_dl vs assertion_jsl*:

|chunks|entities|assertion_dl|assertion_jsl|
|-|-|-|-|
|Mesothelioma|PROBLEM|present|Present|
|CVA|PROBLEM|absent|Absent|
|cancer|PROBLEM|associated_with_someone_else|Family|
|her INR|TEST|present|Planned|
|Amiodarone|TREATMENT|hypothetical|Hypothetical|
|lymphadenopathy|PROBLEM|absent|Absent|
|stage III disease|PROBLEM|possible|Possible|
|IV piggyback|TREATMENT|conditional|Past|


*Example*:

*Python:*
```bash
...
clinical_assertion = AssertionDLModel.pretrained("assertion_jsl", "en", "clinical/models") \
    .setInputCols(["sentence", "ner_chunk", "embeddings"]) \
    .setOutputCol("assertion")

nlpPipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, word_embeddings, clinical_ner, ner_converter, clinical_assertion])
model = nlpPipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

result = model.transform(spark.createDataFrame([["The patient is a 41-year-old and has a nonproductive cough that started last week. She has had right-sided chest pain radiating to her back with fever starting today. She has no nausea. She has a history of pericarditis and pericardectomy in May 2006 and developed cough with right-sided chest pain, and went to an urgent care center and Chest x-ray revealed right-sided pleural effusion. In family history, her father has a colon cancer history."]], ["text"])
```

*Results*:

```
+-------------------+-----+---+-------------------------+-------+---------+
|chunk              |begin|end|ner_label                |sent_id|assertion|
+-------------------+-----+---+-------------------------+-------+---------+
|nonproductive cough|35   |53 |Symptom                  |0      |Present  |
|last week          |68   |76 |RelativeDate             |0      |Past     |
|chest pain         |103  |112|Symptom                  |1      |Present  |
|fever              |141  |145|VS_Finding               |1      |Present  |
|today              |156  |160|RelativeDate             |1      |Present  |
|nausea             |174  |179|Symptom                  |2      |Absent   |
|pericarditis       |203  |214|Disease_Syndrome_Disorder|3      |Past     |
|pericardectomy     |220  |233|Procedure                |3      |Past     |
|May 2006           |238  |245|Date                     |3      |Past     |
|cough              |261  |265|Symptom                  |3      |Past     |
|chest pain         |284  |293|Symptom                  |3      |Past     |
|Chest x-ray        |334  |344|Test                     |3      |Past     |
|pleural effusion   |367  |382|Disease_Syndrome_Disorder|3      |Past     |
|colon cancer       |421  |432|Oncological              |4      |Family   |
+-------------------+-----+---+-------------------------+-------+---------+
```

#### New Sentence Entity Resolver Model

We are releasing `sbiobertresolve_rxnorm_disposition` model that maps medication entities (like drugs/ingredients) to RxNorm codes and their dispositions using `sbiobert_base_cased_mli` Sentence Bert Embeddings. In the result, look for the aux_label parameter in the metadata to get dispositions that were divided by `|`.

*Example*:

*Python*:
```bash

documentAssembler = DocumentAssembler()\
      .setInputCol("text")\
      .setOutputCol("ner_chunk")

sbert_embedder = BertSentenceEmbeddings.pretrained('sbiobert_base_cased_mli', 'en','clinical/models')\
      .setInputCols(["ner_chunk"])\
      .setOutputCol("sbert_embeddings")

rxnorm_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_rxnorm_disposition", "en", "clinical/models") \
      .setInputCols(["ner_chunk", "sbert_embeddings"]) \
      .setOutputCol("rxnorm_code")\
      .setDistanceFunction("EUCLIDEAN")

pipelineModel = PipelineModel(
    stages = [
        documentAssembler,
        sbert_embedder,
        rxnorm_resolver
    ])

rxnorm_lp = LightPipeline(pipelineModel)

result = rxnorm_lp.fullAnnotate("belimumab 80 mg/ml injectable solution")
```

*Results*:

```
|    | chunks                                | code    | resolutions                                                                                                                                                                                 | all_codes                                         | all_k_aux_labels                                                                            | all_distances                                 |
|---:|:--------------------------------------|:--------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------|:--------------------------------------------------------------------------------------------|:----------------------------------------------|
|  0 |belimumab 80 mg/ml injectable solution | 1092440 | [belimumab 80 mg/ml injectable solution, belimumab 80 mg/ml injectable solution [benlysta], ifosfamide 80 mg/ml injectable solution, belimumab 80 mg/ml [benlysta], belimumab 80 mg/ml, ...]| [1092440, 1092444, 107034, 1092442, 1092438, ...] | [Immunomodulator, Immunomodulator, Alkylating agent, Immunomodulator, Immunomodulator, ...] | [0.0000, 0.0145, 0.0479, 0.0619, 0.0636, ...] |
```

#### Finetuning Sentence Entity Resolvers with Your Data

Instead of starting from scratch when training a new Sentence Entity Resolver model, you can train a new model by adding your new data to the pretrained model.

There's a new method `setPretrainedModelPath(path)`, which allows you to point the training process to an existing model, and allows you to initialize your model with the data from the pretrained model.

When both the new data and the pretrained model contain the same code, you will see both of the results at the top.

Here is a sample notebook : [Finetuning Sentence Entity Resolver Model Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/13.1.Finetuning_Sentence_Entity_Resolver_Model.ipynb)

*Example:*

In the example below, we changed the code of `sepsis` to `X1234` and re-retrain the main ICD10-CM model with this new dataset. So we want to see the `X1234` code as a result in the all_codes.

*Python:*
```bash
...
bertExtractor = SentenceEntityResolverApproach()\
  .setNeighbours(50)\
  .setThreshold(1000)\
  .setInputCols("sentence_embeddings")\
  .setNormalizedCol("description_normalized")\   # concept_name
  .setLabelCol("code")\     # concept_code
  .setOutputCol("recognized_code")\
  .setDistanceFunction("EUCLIDEAN")\
  .setCaseSensitive(False)\         
  .setUseAuxLabel(True)\         # if exist  
  .setPretrainedModelPath("path_to_a_pretrained_model")


new_model = bertExtractor.fit("new_dataset")
new_model.save("models/new_resolver_model")  # save and use later
```

```
...
resolver_model = SentenceEntityResolverModel.load("models/new_resolver_model") \
      .setInputCols(["ner_chunk", "sentence_embeddings"]) \
      .setOutputCol("output_code")

pipelineModel = PipelineModel(
    stages = [
        documentAssembler,
        sentence_embedder,
        resolver_model])

light_model = LightPipeline(pipelineModel)
light_model.fullAnnotate("sepsis")
```

*Main Model Results*:

|chunks |begin |end |code |all_codes |resolutions |all_k_aux_labels |all_distances|
|-|-|-|-|-|-|-|-|
|sepsis   |0   |5   |A4189 |[A4189, L419, A419, A267, E771, ...]   | [sepsis [Other specified sepsis], parapsoriasis [Parapsoriasis, unspecified], postprocedural sepsis [Sepsis, unspecified organism], erysipelothrix sepsis [Erysipelothrix sepsis], fucosidosis [Defects in glycoprotein degradation], ... ]| [1\|1\|2, 1\|1\|2, 1\|1\|2, 1\|1\|2, 1\|1\|23, ...]  |[0.0000, 0.2079, 0.2256, 0.2359, 0.2399,...]   |

*Re-Trained Model Results*:

|chunks |begin |end |code |all_codes |resolutions |all_k_aux_labels |all_distances|
|-|-|-|-|-|-|-|-|
|sepsis   |0   |5   |X1234 |[X1234, A4189, A419, L419, A267, ...]   | [sepsis [Sepsis, new resolution], sepsis [Other specified sepsis], SEPSIS [Sepsis, unspecified organism], parapsoriasis [Parapsoriasis, unspecified], erysipelothrix sepsis [Erysipelothrix sepsis], ... ]| [1\|1\|74, 1\|1\|2, 1\|1\|2, 1\|1\|2, 1\|1\|2, ...]  |[0.0000, 0.0000, 0.0000, 0.2079, 0.2359, ...]   |

#### New Clinical NER Models

+ `ner_jsl_slim`: This model is trained based on `ner_jsl` model with more generalized entities.

  (`Death_Entity`, `Medical_Device`, `Vital_Sign`, `Alergen`, `Drug`, `Clinical_Dept`, `Lifestyle`, `Symptom`, `Body_Part`, `Physical_Measurement`, `Admission_Discharge`, `Date_Time`, `Age`, `Birth_Entity`, `Header`, `Oncological`, `Substance_Quantity`, `Test_Result`, `Test`, `Procedure`, `Treatment`, `Disease_Syndrome_Disorder`, `Pregnancy_Newborn`, `Demographics`)

*ner_jsl vs ner_jsl_slim*:

|chunks|ner_jsl|ner_jsl_slim|
|-|-|-|
|Description:|Section_Header|Header|
|atrial fibrillation|Heart_Disease|Disease_Syndrome_Disorder|
|August 24, 2007|Date|Date_Time|
|transpleural fluoroscopy|Procedure|Test|
|last week|RelativeDate|Date_Time|
|She|Gender|Demographics|
|fever|VS_Finding|Vital_Sign|
|PAST MEDICAL HISTORY:|Medical_History_Header|Header|
|Pericardial window|Internal_organ_or_component|Body_Part|
|FAMILY HISTORY:|Family_History_Header|Header|
|CVA|Cerebrovascular_Disease|Disease_Syndrome_Disorder|
|diabetes|Diabetes|Disease_Syndrome_Disorder|
|married|Relationship_Status|Demographics|
|alcohol|Alcohol|Lifestyle|
|illicit drug|Substance|Lifestyle|
|Coumadin|Drug_BrandName|Drug|
|Blood pressure 123/95|Blood_Pressure|Vital_Sign|
|heart rate 83|Pulse|Vital_Sign|
|anticoagulated|Drug_Ingredient|Drug|


*Example*:

*Python*:

```bash
...
embeddings_clinical = WordEmbeddingsModel().pretrained('embeddings_clinical', 'en', 'clinical/models') \
    .setInputCols(['sentence', 'token']) \
    .setOutputCol('embeddings')

clinical_ner = MedicalNerModel.pretrained("ner_jsl_slim", "en", "clinical/models") \
  .setInputCols(["sentence", "token", "embeddings"]) \
  .setOutputCol("ner")
...

nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings_clinical,  clinical_ner, ner_converter])
model = nlpPipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

results = model.transform(spark.createDataFrame([["HISTORY: 30-year-old female presents for digital bilateral mammography secondary to a soft tissue lump palpated by the patient in the upper right shoulder. The patient has a family history of breast cancer within her mother at age 58. Patient denies personal history of breast cancer."]], ["text"]))
```

*Results*:
```bash
|    | chunk            | entity       |
|---:|:-----------------|:-------------|
|  0 | HISTORY:         | Header       |
|  1 | 30-year-old      | Age          |
|  2 | female           | Demographics |
|  3 | mammography      | Test         |
|  4 | soft tissue lump | Symptom      |
|  5 | shoulder         | Body_Part    |
|  6 | breast cancer    | Oncological  |
|  7 | her mother       | Demographics |
|  8 | age 58           | Age          |
|  9 | breast cancer    | Oncological  |
```

+ `ner_jsl_biobert` : This model is the BioBert version of `ner_jsl` model and trained with `biobert_pubmed_base_cased` embeddings.

+ `ner_jsl_greedy_biobert` : This model is the BioBert version of `ner_jsl_greedy` models and trained with `biobert_pubmed_base_cased` embeddings.

*Example*:

*Python*:

```bash
...
embeddings_clinical = BertEmbeddings.pretrained('biobert_pubmed_base_cased') \
    .setInputCols(['sentence', 'token']) \
    .setOutputCol('embeddings')
clinical_ner = MedicalNerModel.pretrained("ner_jsl_greedy_biobert", "en", "clinical/models") \
  .setInputCols(["sentence", "token", "embeddings"]) \
  .setOutputCol("ner")
...
nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings_clinical,  clinical_ner, ner_converter])
model = nlpPipeline.fit(spark.createDataFrame([[""]]).toDF("text"))
results = model.transform(spark.createDataFrame([["The patient is a 21-day-old Caucasian male here for 2 days of congestion - mom has been suctioning yellow discharge from the patient's nares, plus she has noticed some mild problems with his breathing while feeding (but negative for any perioral cyanosis or retractions). One day ago, mom also noticed a tactile temperature and gave the patient Tylenol. Baby also has had some decreased p.o. intake. His normal breast-feeding is down from 20 minutes q.2h. to 5 to 10 minutes secondary to his respiratory congestion. He sleeps well, but has been more tired and has been fussy over the past 2 days. The parents noticed no improvement with albuterol treatments given in the ER. His urine output has also decreased; normally he has 8 to 10 wet and 5 dirty diapers per 24 hours, now he has down to 4 wet diapers per 24 hours. Mom denies any diarrhea. His bowel movements are yellow colored and soft in nature."]], ["text"]))
```

*Results*:

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

#### New CMS-HCC risk-adjustment score calculation module

We are releasing a new module to calculate medical risk adjusment score by using the Centers for Medicare & Medicaid Service (CMS) risk adjustment model. The main input to this model are ICD codes of the diseases. After getting ICD codes of diseases by Spark NLP Healthcare ICD resolvers, risk score can be calculated by this module in spark environment.
Current supported version for the model is CMS-HCC V24.

The model needs following parameters in order to calculate the risk score:
- ICD Codes
- Age
- Gender
- The eligibility segment of the patient
- Original reason for entitlement
- If the patient is in Medicaid or not
- If the patient is disabled or not

*Example*:

*Python:*
```
sample_patients.show()
```
*Results*:
```
+----------+------------------------------+---+------+
|Patient_ID|ICD_codes                     |Age|Gender|
+----------+------------------------------+---+------+
|101       |[E1169, I5030, I509, E852]    |64 |F     |
|102       |[G629, D469, D6181]           |77 |M     |
|103       |[D473, D473, D473, M069, C969]|16 |F     |
+----------+------------------------------+---+------+
```


*Python:*
```
from sparknlp_jsl.functions import profile
df = df.withColumn("hcc_profile", profile(df.ICD_codes, df.Age, df.Gender))

df = df.withColumn("hcc_profile", F.from_json(F.col("hcc_profile"), schema))
df= df.withColumn("risk_score", df.hcc_profile.getItem("risk_score"))\
      .withColumn("hcc_lst", df.hcc_profile.getItem("hcc_map"))\
      .withColumn("parameters", df.hcc_profile.getItem("parameters"))\
      .withColumn("details", df.hcc_profile.getItem("details"))\

df.select('Patient_ID', 'risk_score','ICD_codes', 'Age', 'Gender').show(truncate=False )

df.show(truncate=100, vertical=True)

```

*Results*:
{% raw %}
```

+----------+----------+------------------------------+---+------+
|Patient_ID|risk_score|ICD_codes                     |Age|Gender|
+----------+----------+------------------------------+---+------+
|101       |0.827     |[E1169, I5030, I509, E852]    |64 |F     |
|102       |1.845     |[G629, D469, D6181]           |77 |M     |
|103       |1.288     |[D473, D473, D473, M069, C969]|16 |F     |
+----------+----------+------------------------------+---+------+

RECORD 0-------------------------------------------------------------------------------------------------------------------
 Patient_ID          | 101                                                                                                  
 ICD_codes           | [E1169, I5030, I509, E852]                                                                           
 Age                 | 64                                                                                                   
 Gender              | F                                                                                                    
 Eligibility_Segment | CNA                                                                                                  
 OREC                | 0                                                                                                    
 Medicaid            | false                                                                                                 
 Disabled            | false                                                                                                
 hcc_profile         | {{"CNA_HCC18":0.302,"CNA_HCC85":0.331,"CNA_HCC23":0.194,"CNA_D3":0.0,"CNA_HCC85_gDiabetesMellit":...
 risk_score          | 0.827                                                                                                
 hcc_lst             | {"E1169":["HCC18"],"I5030":["HCC85"],"I509":["HCC85"],"E852":["HCC23"]}                              
 parameters          | {"elig":"CNA","age":64,"sex":"F","origds":'0',"disabled":false,"medicaid":false}                   
 details             | {"CNA_HCC18":0.302,"CNA_HCC85":0.331,"CNA_HCC23":0.194,"CNA_D3":0.0,"CNA_HCC85_gDiabetesMellit":0.0}
-RECORD 1-------------------------------------------------------------------------------------------------------------------
 Patient_ID          | 102                                                                                                  
 ICD_codes           | [G629, D469, D6181]                                                                                  
 Age                 | 77                                                                                                   
 Gender              | M                                                                                                    
 Eligibility_Segment | CNA                                                                                                  
 OREC                | 0                                                                                                    
 Medicaid            | false                                                                                                 
 Disabled            | false                                                                                                 
 hcc_profile         | {{"CNA_M75_79":0.473,"CNA_D1":0.0,"CNA_HCC46":1.372}, ["D1","HCC46"], {"D469":["HCC46"]}, {"elig"...
 risk_score          | 1.845                                                                                                
 hcc_lst             | {"D469":["HCC46"]}                                                                                   
 parameters          | {"elig":"CNA","age":77,"sex":"M","origds":'0',"disabled":false,"medicaid":false}                   
 details             | {"CNA_M75_79":0.473,"CNA_D1":0.0,"CNA_HCC46":1.372}                                                  
-RECORD 2-------------------------------------------------------------------------------------------------------------------
 Patient_ID          | 103                                                                                                  
 ICD_codes           | [D473, D473, D473, M069, C969]                                                                       
 Age                 | 16                                                                                                   
 Gender              | F                                                                                                    
 Eligibility_Segment | CNA                                                                                                  
 OREC                | 0                                                                                                    
 Medicaid            | false                                                                                                
 Disabled            | false                                                                                                
 hcc_profile         | {{"CNA_HCC10":0.675,"CNA_HCC40":0.421,"CNA_HCC48":0.192,"CNA_D3":0.0}, ["HCC10","HCC40","HCC48","...
 risk_score          | 1.288                                                                                                
 hcc_lst             | {"D473":["HCC48"],"M069":["HCC40"],"C969":["HCC10"]}                                                 
 parameters          | {"elig":"CNA","age":16,"sex":"F","origds":'0',"disabled":false,"medicaid":false}                   
 details             | {"CNA_HCC10":0.675,"CNA_HCC40":0.421,"CNA_HCC48":0.192,"CNA_D3":0.0}
```
{% endraw %}

Here is a sample notebook : [Calculating Medicare Risk Adjustment Score](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.1.Calculate_Medicare_Risk_Adjustment_Score.ipynb)

#### New Embedding generation module for entity resolution

We are releasing a new annotator `BertSentenceChunkEmbeddings` to let users aggregate sentence embeddings and ner chunk embeddings to get more specific and accurate resolution codes. It works by averaging context and chunk embeddings to get contextual information. This is specially helpful when ner chunks do not have additional information (like body parts or severity) as explained in the example below. Input to this annotator is the context (sentence) and ner chunks, while the output is embedding for each chunk that can be fed to the resolver model. The `setChunkWeight` parameter can be used to control the influence of surrounding context. Example below shows the comparison of old vs new approach.

Sample Notebook: [Improved_Entity_Resolution_with_SentenceChunkEmbeddings](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/24.1.Improved_Entity_Resolution_with_SentenceChunkEmbeddings.ipynb)

*Example*:

*Python:*
```
...
sentence_chunk_embeddings = BertSentenceChunkEmbeddings\
    .pretrained("sbiobert_base_cased_mli", "en", "clinical/models")\
    .setInputCols(["sentences", "ner_chunk"])\
    .setOutputCol("sentence_chunk_embeddings")\
    .setChunkWeight(0.5)

resolver = SentenceEntityResolverModel.pretrained('sbiobertresolve_icd10cm', 'en', 'clinical/models')\
            .setInputCols(["ner_chunk", "sentence_chunk_embeddings"]) \
              .setOutputCol("resolution")

text = """A 20 year old female patient badly tripped while going down stairs. She complains of right leg pain.
Her x-ray showed right hip fracture. Hair line fractures also seen on the left knee joint.
She also suffered from trauma and slight injury on the head.

OTHER CONDITIONS: She was also recently diagnosed with diabetes, which is of type 2.
"""

nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings_clinical,  clinical_ner, ner_converter, sentence_chunk_embeddings, resolver])
model = nlpPipeline.fit(spark.createDataFrame([[""]]).toDF("text"))
results = model.transform(spark.createDataFrame([[text]], ["text"]))
```

*Results*:
```
|    | chunk               | entity              | code_with_old_approach | resolutions_with_old_approach                              | code_with_new_approach | resolutions_with_new_approach                                                                 |
|---:|:--------------------|:--------------------|:-----------------------|:-----------------------------------------------------------|:-----------------------|:----------------------------------------------------------------------------------------------|
|  0 | leg pain            | Symptom             | R1033                  | Periumbilical pain                                         | M79661                 | Pain in right lower leg                                                                       |
|  1 | hip fracture        | Injury_or_Poisoning | M84459S                | Pathological fracture, hip, unspecified, sequela           | M84451S                | Pathological fracture, right femur, sequela                                                   |
|  2 | Hair line fractures | Injury_or_Poisoning | S070XXS                | Crushing injury of face, sequela                           | S92592P                | Other fracture of left lesser toe(s), subsequent encounter for fracture with malunion         |
|  3 | trauma              | Injury_or_Poisoning | T794XXS                | Traumatic shock, sequela                                   | S0083XS                | Contusion of other part of head, sequela                                                      |
|  4 | slight injury       | Injury_or_Poisoning | B03                    | Smallpox                                                   | S0080XD                | Unspecified superficial injury of other part of head, subsequent encounter                    |
|  5 | diabetes            | Diabetes            | E118                   | Type 2 diabetes mellitus with unspecified complications    | E1169                  | Type 2 diabetes mellitus with other specified complication                                    |
```

**To see more, please check :** [Spark NLP Healthcare Workshop Repo](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings/Healthcare)

<div class="prev_ver h3-box" markdown="1">

## Versions

</div>

<ul class="pagination">
    <li>
        <a href="release_notes_3_1_3">Version 3.1.3</a>
    </li>
    <li>
        <strong>Version 3.2.0</strong>
    </li>
    <li>
        <a href="release_notes_3_2_1">Version 3.2.1</a>
    </li>
</ul>

<ul class="pagination owl-carousel pagination_big">
    <li><a href="release_notes_4_1_0">4.1.0</a></li>
    <li><a href="release_notes_4_0_2">4.0.2</a></li>
    <li><a href="release_notes_4_0_0">4.0.0</a></li>
    <li><a href="release_notes_3_5_3">3.5.3</a></li>
    <li><a href="release_notes_3_5_2">3.5.2</a></li>
    <li><a href="release_notes_3_5_1">3.5.1</a></li>
    <li><a href="release_notes_3_5_0">3.5.0</a></li>
    <li><a href="release_notes_3_4_2">3.4.2</a></li>
    <li><a href="release_notes_3_4_1">3.4.1</a></li>
    <li><a href="release_notes_3_4_0">3.4.0</a></li>
    <li><a href="release_notes_3_3_4">3.3.4</a></li>
    <li><a href="release_notes_3_3_2">3.3.2</a></li>
    <li><a href="release_notes_3_3_1">3.3.1</a></li>
    <li><a href="release_notes_3_3_0">3.3.0</a></li>
    <li><a href="release_notes_3_2_3">3.2.3</a></li>
    <li><a href="release_notes_3_2_2">3.2.2</a></li>
    <li><a href="release_notes_3_2_1">3.2.1</a></li>
    <li class="active"><a href="release_notes_3_2_0">3.2.0</a></li>
    <li><a href="release_notes_3_1_3">3.1.3</a></li>
    <li><a href="release_notes_3_1_2">3.1.2</a></li>
    <li><a href="release_notes_3_1_1">3.1.1</a></li>
    <li><a href="release_notes_3_1_0">3.1.0</a></li>
    <li><a href="release_notes_3_0_3">3.0.3</a></li>
    <li><a href="release_notes_3_0_2">3.0.2</a></li>
    <li><a href="release_notes_3_0_1">3.0.1</a></li>
    <li><a href="release_notes_3_0_0">3.0.0</a></li>
    <li><a href="release_notes_2_7_6">2.7.6</a></li>
    <li><a href="release_notes_2_7_5">2.7.5</a></li>
    <li><a href="release_notes_2_7_4">2.7.4</a></li>
    <li><a href="release_notes_2_7_3">2.7.3</a></li>
    <li><a href="release_notes_2_7_2">2.7.2</a></li>
    <li><a href="release_notes_2_7_1">2.7.1</a></li>
    <li><a href="release_notes_2_7_0">2.7.0</a></li>
    <li><a href="release_notes_2_6_2">2.6.2</a></li>
    <li><a href="release_notes_2_6_0">2.6.0</a></li>
    <li><a href="release_notes_2_5_5">2.5.5</a></li>
    <li><a href="release_notes_2_5_3">2.5.3</a></li>
    <li><a href="release_notes_2_5_2">2.5.2</a></li>
    <li><a href="release_notes_2_5_0">2.5.0</a></li>
    <li><a href="release_notes_2_4_6">2.4.6</a></li>
    <li><a href="release_notes_2_4_5">2.4.5</a></li>
    <li><a href="release_notes_2_4_2">2.4.2</a></li>
    <li><a href="release_notes_2_4_1">2.4.1</a></li>
    <li><a href="release_notes_2_4_0">2.4.0</a></li>
</ul>