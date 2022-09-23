---
layout: docs
header: true
seotitle: Spark NLP for Healthcare | John Snow Labs
title: Spark NLP for Healthcare Release Notes 3.3.4
permalink: /docs/en/spark_nlp_healthcare_versions/release_notes_3_3_4
key: docs-licensed-release-notes
modify_date: 2021-07-14
show_nav: true
sidebar:
    nav: sparknlp-healthcare
---

## 3.3.4
We are glad to announce that Spark NLP Healthcare 3.3.4 has been released!



#### Highlights

+ New Clinical NER Models
+ New NER Model Finder Pretrained Pipeline
+ New Relation Extraction Model
+ New LOINC, MeSH, NDC and SNOMED Entity Resolver Models
+ Updated RxNorm Sentence Entity Resolver Model
+ New Shift Days Feature in StructuredDeid Deidentification Module
+ New Multiple Chunks Merge Ability in ChunkMergeApproach
+ New setBlackList Feature in ChunkMergeApproach
+ New setBlackList Feature in NerConverterInternal
+ New setLabelCasing Feature in MedicalNerModel
+ New Update Models Functionality
+ New and Updated Notebooks

#### New Clinical NER Models

We have three new clinical NER models.

+ `ner_deid_subentity_augmented_i2b2` : This model annotates text to find protected health information(PHI) that may need to be removed. It is trained with 2014 i2b2 dataset (no augmentation applied) and can detect `MEDICALRECORD`, `ORGANIZATION`, `DOCTOR`, `USERNAME`, `PROFESSION`, `HEALTHPLAN`, `URL`, `CITY`, `DATE`, `LOCATION-OTHER`, `STATE`, `PATIENT`, `DEVICE`, `COUNTRY`, `ZIP`, `PHONE`, `HOSPITAL`, `EMAIL`, `IDNUM`, `SREET`, `BIOID`, `FAX`, `AGE` entities.

*Example* :

```bash
...
deid_ner = MedicalNerModel.pretrained("ner_deid_subentity_augmented_i2b2", "en", "clinical/models") \
      .setInputCols(["sentence", "token", "embeddings"]) \
      .setOutputCol("ner")
...

results = ner_model.transform(spark.createDataFrame([["A. Record date : 2093-01-13, David Hale, M.D., Name : Hendrickson, Ora MR. # 7194334 Date : 01/13/93 PCP : Oliveira, 25 years old, Record date : 1-11-2000. Cocke County Baptist Hospital. 0295 Keats Street. Phone +1 (302) 786-5227. Patient's complaints first surfaced when he started working for Brothers Coal-Mine."]], ["text"]))
```

*Results* :

```bash
+-----------------------------+-------------+
|chunk                        |ner_label    |
+-----------------------------+-------------+
|2093-01-13                   |DATE         |
|David Hale                   |DOCTOR       |
|Hendrickson, Ora             |PATIENT      |
|7194334                      |MEDICALRECORD|
|01/13/93                     |DATE         |
|Oliveira                     |DOCTOR       |
|25                           |AGE          |
|1-11-2000                    |DATE         |
|Cocke County Baptist Hospital|HOSPITAL     |
|0295 Keats Street            |STREET       |
|(302) 786-5227               |PHONE        |
|Brothers Coal-Mine Corp      |ORGANIZATION |
+-----------------------------+-------------+
```

+ `ner_biomarker` : This model is trained to extract biomarkers, therapies, oncological, and other general concepts from text. Following are the entities it can detect: `Oncogenes`, `Tumor_Finding`, `UnspecificTherapy`, `Ethnicity`, `Age`, `ResponseToTreatment`, `Biomarker`, `HormonalTherapy`, `Staging`, `Drug`, `CancerDx`, `Radiotherapy`, `CancerSurgery`, `TargetedTherapy`, `PerformanceStatus`, `CancerModifier`, `Radiological_Test_Result`, `Biomarker_Measurement`, `Metastasis`, `Radiological_Test`, `Chemotherapy`, `Test`, `Dosage`, `Test_Result`, `Immunotherapy`, `Date`, `Gender`, `Prognostic_Biomarkers`, `Duration`, `Predictive_Biomarkers`

*Example* :

```bash
...
clinical_ner = MedicalNerModel.pretrained("ner_biomarker", "en", "clinical/models") \
  .setInputCols(["sentence", "token", "embeddings"]) \
  .setOutputCol("ner")
...

results = ner_model.transform(spark.createDataFrame([["Here , we report the first case of an intraductal tubulopapillary neoplasm of the pancreas with clear cell morphology . Immunohistochemistry revealed positivity for Pan-CK , CK7 , CK8/18 , MUC1 , MUC6 , carbonic anhydrase IX , CD10 , EMA , β-catenin and e-cadherin ."]], ["text"]))
```

*Results* :

```bash
|    | ner_chunk                | entity                |   confidence |
|---:|:-------------------------|:----------------------|-------------:|
|  0 | intraductal              | CancerModifier        |     0.9934   |
|  1 | tubulopapillary          | CancerModifier        |     0.6403   |
|  2 | neoplasm of the pancreas | CancerDx              |     0.758825 |
|  3 | clear cell               | CancerModifier        |     0.9633   |
|  4 | Immunohistochemistry     | Test                  |     0.9534   |
|  5 | positivity               | Biomarker_Measurement |     0.8795   |
|  6 | Pan-CK                   | Biomarker             |     0.9975   |
|  7 | CK7                      | Biomarker             |     0.9975   |
|  8 | CK8/18                   | Biomarker             |     0.9987   |
|  9 | MUC1                     | Biomarker             |     0.9967   |
| 10 | MUC6                     | Biomarker             |     0.9972   |
| 11 | carbonic anhydrase IX    | Biomarker             |     0.937567 |
| 12 | CD10                     | Biomarker             |     0.9974   |
| 13 | EMA                      | Biomarker             |     0.9899   |
| 14 | β-catenin                | Biomarker             |     0.8059   |
| 15 | e-cadherin               | Biomarker             |     0.9806   |
```

+ `ner_nihss` : NER model that can identify entities according to NIHSS guidelines for clinical stroke assessment to evaluate neurological status in acute stroke patients. Here are the labels it can detect : `11_ExtinctionInattention`, `6b_RightLeg`, `1c_LOCCommands`, `10_Dysarthria`, `NIHSS`, `5_Motor`, `8_Sensory`, `4_FacialPalsy`, `6_Motor`, `2_BestGaze`, `Measurement`, `6a_LeftLeg`, `5b_RightArm`, `5a_LeftArm`, `1b_LOCQuestions`, `3_Visual`, `9_BestLanguage`, `7_LimbAtaxia`, `1a_LOC` .

*Example* :

```bash
...
clinical_ner = MedicalNerModel.pretrained("ner_nihss", "en", "clinical/models") \
  .setInputCols(["sentence", "token", "embeddings"]) \
  .setOutputCol("ner")
...

results = ner_model.transform(spark.createDataFrame([["Abdomen , soft , nontender . NIH stroke scale on presentation was 23 to 24 for , one for consciousness , two for month and year and two for eye / grip , one to two for gaze , two for face , eight for motor , one for limited ataxia , one to two for sensory , three for best language and two for attention . On the neurologic examination the patient was intermittently"]], ["text"]))
```  

*Results* :

```bash
|    | chunk              | entity                   |
|---:|:-------------------|:-------------------------|
|  0 | NIH stroke scale   | NIHSS                    |
|  1 | 23 to 24           | Measurement              |
|  2 | one                | Measurement              |
|  3 | consciousness      | 1a_LOC                   |
|  4 | two                | Measurement              |
|  5 | month and year and | 1b_LOCQuestions          |
|  6 | two                | Measurement              |
|  7 | eye / grip         | 1c_LOCCommands           |
|  8 | one to             | Measurement              |
|  9 | two                | Measurement              |
| 10 | gaze               | 2_BestGaze               |
| 11 | two                | Measurement              |
| 12 | face               | 4_FacialPalsy            |
| 13 | eight              | Measurement              |
| 14 | one                | Measurement              |
| 15 | limited            | 7_LimbAtaxia             |
| 16 | ataxia             | 7_LimbAtaxia             |
| 17 | one to two         | Measurement              |
| 18 | sensory            | 8_Sensory                |
| 19 | three              | Measurement              |
| 20 | best language      | 9_BestLanguage           |
| 21 | two                | Measurement              |
| 22 | attention          | 11_ExtinctionInattention |
```

#### New NER Model Finder Pretrained Pipeline

We are releasing new `ner_model_finder` pretrained pipeline trained with bert embeddings that can be used to find the most appropriate NER model given the entity name.

*Example* :

```bash
from sparknlp.pretrained import PretrainedPipeline
finder_pipeline = PretrainedPipeline("ner_model_finder", "en", "clinical/models")

result = finder_pipeline.fullAnnotate("psychology")
```

*Results* :

|entity|top models|all models|resolutions|
|-|-|-|-|
|psychology|['ner_medmentions_coarse', 'jsl_rd_ner_wip_greedy_clinical', 'ner_jsl_enriched', 'ner_jsl', 'jsl_ner_wip_modifier_clinical', 'ner_jsl_greedy']  |['ner_medmentions_coarse', 'jsl_rd_ner_wip_greedy_clinical', 'ner_jsl_enriched', 'ner_jsl', 'jsl_ner_wip_modifier_clinical', 'ner_jsl_greedy']:::['jsl_rd_ner_wip_greedy_clinical', 'ner_jsl_enriched', 'ner_jsl_slim', 'ner_jsl', 'jsl_ner_wip_modifier_clinical,...|psychological condition:::clinical department::: ... |

#### New Relation Extraction Model

We are releasing new `redl_nihss_biobert ` relation extraction model that can relate scale items and their measurements according to NIHSS guidelines.

*Example* :

```bash
...
re_model = RelationExtractionDLModel()\
    .pretrained('redl_nihss_biobert', 'en', "clinical/models") \
    .setPredictionThreshold(0.5)\
    .setInputCols(["re_ner_chunks", "sentences"]) \
    .setOutputCol("relations")
...

sample_text = "There , her initial NIHSS score was 4 , as recorded by the ED physicians . This included 2 for weakness in her left leg and 2 for what they felt was subtle ataxia in her left arm and leg ."
result = re_model.transform(spark.createDataFrame([[sample_text]]).toDF("text"))
```

*Results* :

```bash
| chunk1                                | entity1      |   entity1_begin |   entity1_end | entity2     |   chunk2 |   entity2_begin |   entity2_end | relation   |
|:--------------------------------------|:-------------|----------------:|--------------:|:------------|---------:|----------------:|--------------:|:-----------|
| initial NIHSS score                   | NIHSS        |              12 |            30 | Measurement |        4 |              36 |            36 | Has_Value  |
| left leg                              | 6a_LeftLeg   |             111 |           118 | Measurement |        2 |              89 |            89 | Has_Value  |
| subtle ataxia in her left arm and leg | 7_LimbAtaxia |             149 |           185 | Measurement |        2 |             124 |           124 | Has_Value  |
| left leg                              | 6a_LeftLeg   |             111 |           118 | Measurement |        4 |              36 |            36 | 0          |
| initial NIHSS score                   | NIHSS        |              12 |            30 | Measurement |        2 |             124 |           124 | 0          |
| subtle ataxia in her left arm and leg | 7_LimbAtaxia |             149 |           185 | Measurement |        4 |              36 |            36 | 0          |
| subtle ataxia in her left arm and leg | 7_LimbAtaxia |             149 |           185 | Measurement |        2 |              89 |            89 | 0          |
```

#### New LOINC, MeSH, NDC and SNOMED Entity Resolver Models

We have four new Sentence Entity Resolver Models.

+ `sbiobertresolve_mesh` : This model maps clinical entities to Medical Subject Heading (MeSH) codes using `sbiobert_base_cased_mli` Sentence Bert Embeddings.

*Example* :

```bash
...
mesh_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_mesh", "en", "clinical/models") \
      .setInputCols(["ner_chunk", "sentence_embeddings"]) \
      .setOutputCol("mesh_code")\
      .setDistanceFunction("EUCLIDEAN")\
      .setCaseSensitive(False)

...

sample_text = """She was admitted to the hospital with chest pain and found to have bilateral pleural effusion, the right greater than the left. We reviewed the pathology obtained from the pericardectomy in March 2006, which was diagnostic of mesothelioma. At this time, chest tube placement for drainage of the fluid occurred and thoracoscopy with fluid biopsies, which were performed, which revealed malignant mesothelioma."""
result = resolver_model.transform(spark.createDataFrame([[sample_text]]).toDF("text"))
```

*Results* :

```bash
+--------------------------+---------+----------+----------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------+
|                 ner_chunk|   entity| mesh_code|                                                                                           all_codes|                                                                                         resolutions|                                                                                           distances|
+--------------------------+---------+----------+----------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------+
|                chest pain|  PROBLEM|   D002637|D002637:::D059350:::D019547:::D020069:::D015746:::D000072716:::D005157:::D059265:::D001416:::D048...|Chest Pain:::Chronic Pain:::Neck Pain:::Shoulder Pain:::Abdominal Pain:::Cancer Pain:::Facial Pai...|0.0000:::0.0577:::0.0587:::0.0601:::0.0658:::0.0704:::0.0712:::0.0741:::0.0766:::0.0778:::0.0794:...|
|bilateral pleural effusion|  PROBLEM|   D010996|D010996:::D010490:::D011654:::D016724:::D010995:::D016066:::D011001:::D007819:::D035422:::D004653...|Pleural Effusion:::Pericardial Effusion:::Pulmonary Edema:::Empyema, Pleural:::Pleural Diseases::...|0.0309:::0.1010:::0.1115:::0.1213:::0.1218:::0.1398:::0.1425:::0.1401:::0.1451:::0.1464:::0.1464:...|
|             the pathology|     TEST|   D010336|D010336:::D010335:::D001004:::D020969:::C001675:::C536472:::D004194:::D003951:::D013631:::C535329...|Pathology:::Pathologic Processes:::Anus Diseases:::Disease Attributes:::malformins:::Upington dis...|0.0788:::0.0977:::0.1364:::0.1396:::0.1419:::0.1459:::0.1418:::0.1393:::0.1514:::0.1541:::0.1491:...|
|        the pericardectomy|TREATMENT|   D010492|D010492:::D011670:::D018700:::D020884:::D011672:::D005927:::D064727:::D002431:::C000678968:::D011...|Pericardiectomy:::Pulpectomy:::Pleurodesis:::Colpotomy:::Pulpotomy:::Glossectomy:::Posterior Caps...|0.1098:::0.1448:::0.1801:::0.1852:::0.1871:::0.1923:::0.1901:::0.2023:::0.2075:::0.2010:::0.1996:...|
|              mesothelioma|  PROBLEM|D000086002|D000086002:::C535700:::D009208:::D032902:::D018301:::D018199:::C562740:::C000686536:::D018276:::D...|Mesothelioma, Malignant:::Malignant mesenchymal tumor:::Myoepithelioma:::Ganoderma:::Neoplasms, M...|0.0813:::0.1515:::0.1599:::0.1810:::0.1864:::0.1881:::0.1907:::0.1938:::0.1924:::0.1876:::0.2040:...|
|      chest tube placement|TREATMENT|   D015505|D015505:::D019616:::D013896:::D012124:::D013906:::D013510:::D020708:::D035423:::D013903:::D000066...|Chest Tubes:::Thoracic Surgical Procedures:::Thoracic Diseases:::Respiratory Care Units:::Thoraco...|0.0557:::0.1473:::0.1598:::0.1604:::0.1725:::0.1651:::0.1795:::0.1760:::0.1804:::0.1846:::0.1883:...|
|     drainage of the fluid|TREATMENT|   D004322|D004322:::D018495:::C045413:::D021061:::D045268:::D018508:::D005441:::D015633:::D014906:::D001834...|Drainage:::Fluid Shifts:::Bonain's liquid:::Liquid Ventilation:::Flowmeters:::Water Purification:...|0.1141:::0.1403:::0.1582:::0.1549:::0.1586:::0.1626:::0.1599:::0.1655:::0.1667:::0.1656:::0.1741:...|
|              thoracoscopy|TREATMENT|   D013906|D013906:::D020708:::D035423:::D013905:::D035441:::D013897:::D001468:::D000069258:::D013909:::D013...|Thoracoscopy:::Thoracoscopes:::Thoracic Cavity:::Thoracoplasty:::Thoracic Wall:::Thoracic Duct:::...|0.0000:::0.0359:::0.0744:::0.1007:::0.1070:::0.1143:::0.1186:::0.1257:::0.1228:::0.1356:::0.1354:...|
|            fluid biopsies|     TEST|D000073890|D000073890:::D010533:::D020420:::D011677:::D017817:::D001706:::D005441:::D005751:::D013582:::D000...|Liquid Biopsy:::Peritoneal Lavage:::Cyst Fluid:::Punctures:::Nasal Lavage Fluid:::Biopsy:::Fluids...|0.1408:::0.1612:::0.1763:::0.1744:::0.1744:::0.1810:::0.1744:::0.1828:::0.1896:::0.1909:::0.1950:...|
|    malignant mesothelioma|  PROBLEM|D000086002|D000086002:::C535700:::C562740:::D009236:::D007890:::D012515:::D009208:::C009823:::C000683999:::C...|Mesothelioma, Malignant:::Malignant mesenchymal tumor:::Hemangiopericytoma, Malignant:::Myxosarco...|0.0737:::0.1106:::0.1658:::0.1627:::0.1660:::0.1639:::0.1728:::0.1676:::0.1791:::0.1843:::0.1849:...|
+-------+--------------------------+---------+----------+----------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------+
```

+ `sbiobertresolve_ndc` : This model maps clinical entities and concepts (like drugs/ingredients) to [National Drug Codes](https://www.fda.gov/drugs/drug-approvals-and-databases/national-drug-code-directory) using `sbiobert_base_cased_mli` Sentence Bert Embeddings. Also, if a drug has more than one NDC code, it returns all available codes in the all_k_aux_label column separated by `|` symbol.

*Example* :

```bash
...
ndc_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_ndc", "en", "clinical/models") \
      .setInputCols(["ner_chunk", "sentence_embeddings"]) \
      .setOutputCol("ndc_code")\
      .setDistanceFunction("EUCLIDEAN")\
      .setCaseSensitive(False)
...

sample_text = """The patient was transferred secondary to inability and continue of her diabetes, the sacral decubitus, left foot pressure wound, and associated complications of diabetes.
She is given aspirin 81 mg, folic acid 1 g daily, insulin glargine 100 UNT/ML injection and metformin 500 mg p.o. p.r.n."""
result = resolver_model.transform(spark.createDataFrame([[sample_text]]).toDF("text"))
```

*Results* :

```bash
+-------------------------------------+------+-----------+------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|                            ner_chunk|entity|   ndc_code|                                                                   description|                                                                                                                                                                                               all_codes|                                                                                                                                                                                         all_resolutions|                                                                                                                                                                                         other ndc codes|
+-------------------------------------+------+-----------+------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|                        aspirin 81 mg|  DRUG|73089008114|                               aspirin 81 mg/81mg, 81 mg in 1 carton , capsule|[73089008114, 71872708704, 71872715401, 68210101500, 69536028110, 63548086706, 71679001000, 68196090051, 00113400500, 69536018112, 73089008112, 63981056362, 63739043402, 63548086705, 00113046708, 7...|[aspirin 81 mg/81mg, 81 mg in 1 carton , capsule, aspirin 81 mg 81 mg/1, 4 blister pack in 1 bag , tablet, aspirin 81 mg/1, 1 blister pack in 1 bag , tablet, coated, aspirin 81 mg/1, 1 bag in 1 dru...|         [-, -, -, -, -, -, -, -, -, -, -, 63940060962, -, -, -, -, -, -, -, -, 70000042002|00363021879|41250027408|36800046708|59779027408|49035027408|71476010131|81522046708|30142046708, -, -, -, -]|
|                       folic acid 1 g|  DRUG|43744015101|                                   folic acid 1 g/g, 1 g in 1 package , powder|[43744015101, 63238340000, 66326050555, 51552041802, 51552041805, 63238340001, 81919000204, 51552041804, 66326050556, 51552106301, 51927003300, 71092997701, 51927296300, 51552146602, 61281900002, 6...|[folic acid 1 g/g, 1 g in 1 package , powder, folic acid 1 kg/kg, 1 kg in 1 bottle , powder, folic acid 1 kg/kg, 1 kg in 1 drum , powder, folic acid 1 g/g, 5 g in 1 container , powder, folic acid 1...|                                                                                               [-, -, -, -, -, -, -, -, -, -, -, 51552139201, -, -, -, 81919000203, -, 81919000201, -, -, -, -, -, -, -]|
|insulin glargine 100 UNT/ML injection|  DRUG|00088502101|insulin glargine 100 [iu]/ml, 1 vial, glass in 1 package , injection, solution|[00088502101, 00088222033, 49502019580, 00002771563, 00169320111, 00088250033, 70518139000, 00169266211, 50090127600, 50090407400, 00002771559, 00002772899, 70518225200, 70518138800, 00024592410, 0...|[insulin glargine 100 [iu]/ml, 1 vial, glass in 1 package , injection, solution, insulin glargine 100 [iu]/ml, 1 vial, glass in 1 carton , injection, solution, insulin glargine 100 [iu]/ml, 1 vial ...|[-, -, -, 00088221900, -, -, 50090139800|00088502005, -, 70518146200|00169368712, 00169368512|73070020011, 00088221905|49502019675|50090406800, -, 73070010011|00169750111|50090495500, 66733077301|0...|
|                     metformin 500 mg|  DRUG|70010006315|               metformin hydrochloride 500 mg/500mg, 500 mg in 1 drum , tablet|[70010006315, 62207041613, 71052050750, 62207049147, 71052091050, 25000010197, 25000013498, 25000010198, 71052063005, 51662139201, 70010049118, 70882012456, 71052011005, 71052065905, 71052050850, 1...|[metformin hydrochloride 500 mg/500mg, 500 mg in 1 drum , tablet, metformin hcl 500 mg/kg, 50 kg in 1 drum , powder, 5-fluorouracil 500 g/500g, 500 g in 1 container , powder, metformin er 500 mg 50...|                                                                                             [-, -, -, 70010049105, -, -, -, -, -, -, -, -, -, -, -, 71800000801|42571036007, -, -, -, -, -, -, -, -, -]|
+-------------------------------------+------+-----------+------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
```

+ `sbiobertresolve_loinc_augmented` : This model maps extracted clinical NER entities to LOINC codes using `sbiobert_base_cased_mli` Sentence Bert Embeddings. It is trained on the augmented version of the dataset which is used in previous LOINC resolver models.

*Example* :

```bash
...
loinc_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_loinc_augmented","en", "clinical/models") \
     .setInputCols(["ner_chunk", "sentence_embeddings"]) \
     .setOutputCol("loinc_code")\
     .setDistanceFunction("EUCLIDEAN")\
     .setCaseSensitive(False)
...

sample_text="""The patient is a 22-year-old female with a history of obesity. She has a Body mass index (BMI) of 33.5 kg/m2, aspartate aminotransferase 64, and alanine aminotransferase 126. Her hgba1c is 8.2%."""
result = resolver_model.transform(spark.createDataFrame([[sample_text]]).toDF("text"))
```

*Results* :

```bash
+--------------------------+-----+---+------+----------+--------------------------------------------------+--------------------------------------------------+
|                     chunk|begin|end|entity|Loinc_Code|                                         all_codes|                                       resolutions|
+--------------------------+-----+---+------+----------+--------------------------------------------------+--------------------------------------------------+
|           Body mass index|   74| 88|  Test| LP35925-4|LP35925-4:::BDYCRC:::LP172732-2:::39156-5:::LP7...|body mass index:::body circumference:::body mus...|
|aspartate aminotransferase|  111|136|  Test| LP15426-7|LP15426-7:::14409-7:::LP307348-5:::LP15333-5:::...|aspartate aminotransferase::: aspartate transam...|
|  alanine aminotransferase|  146|169|  Test| LP15333-5|LP15333-5:::LP307326-1:::16324-6:::LP307348-5::...|alanine aminotransferase:::alanine aminotransfe...|
|                    hgba1c|  180|185|  Test|   17855-8|17855-8:::4547-6:::55139-0:::72518-4:::45190-6:...| hba1c::: hgb a1::: hb1::: hcds1::: hhc1::: htr...|
+--------------------------+-----+---+------+----------+--------------------------------------------------+--------------------------------------------------+
```

+ `sbiobertresolve_clinical_snomed_procedures_measurements` : This model maps medical entities to SNOMED codes using `sent_biobert_clinical_base_cased` Sentence Bert Embeddings. The corpus of this model includes `Procedures` and `Measurement` domains.

*Example* :

```bash
...
snomed_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_clinical_snomed_procedures_measurements", "en", "clinical/models") \
      .setInputCols(["ner_chunk", "sbert_embeddings"]) \
      .setOutputCol("snomed_code")
...

light_model = LightPipeline(resolver_model)
result = light_model.fullAnnotate(['coronary calcium score', 'heart surgery', 'ct scan', 'bp value'])

```

*Results* :

```bash
|    | chunk                  |      code | code_description              | all_k_codes                                                                     | all_k_resolutions                                                                                                                                               |
|---:|:-----------------------|----------:|:------------------------------|:--------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------|
|  0 | coronary calcium score | 450360000 | Coronary artery calcium score | ['450360000', '450734004', '1086491000000104', '1086481000000101', '762241007'] | ['Coronary artery calcium score', 'Coronary artery calcium score', 'Dundee Coronary Risk Disk score', 'Dundee Coronary Risk rank', 'Dundee Coronary Risk Disk'] |
|  1 | heart surgery          |   2598006 | Open heart surgery            | ['2598006', '64915003', '119766003', '34068001', '233004008']                   | ['Open heart surgery', 'Operation on heart', 'Heart reconstruction', 'Heart valve replacement', 'Coronary sinus operation']                                     |
|  2 | ct scan                | 303653007 | CT of head                    | ['303653007', '431864000', '363023007', '418272005', '241577003']               | ['CT of head', 'CT guided injection', 'CT of site', 'CT angiography', 'CT of spine']                                                                            |
|  3 | bp value               |  75367002 | Blood pressure                | ['75367002', '6797001', '723232008', '46973005', '427732000']                   | ['Blood pressure', 'Mean blood pressure', 'Average blood pressure', 'Blood pressure taking', 'Speed of blood pressure response']                                |
```

#### Updated RxNorm Sentence Entity Resolver Model

We have updated `sbiobertresolve_rxnorm_augmented` model training on an augmented version of the dataset used in previous versions of the model.

#### New Shift Days Feature in StructuredDeid Deidentification Module

 Now we can shift n days in the structured deidentification when the column is a Date.

 *Example* :

 ```pyhton
 df = spark.createDataFrame([
            ["Juan García", "13/02/1977", "711 Nulla St.", "140", "673 431234"],
            ["Will Smith", "23/02/1977", "1 Green Avenue.", "140", "+23 (673) 431234"],
            ["Pedro Ximénez", "11/04/1900", "Calle del Libertador, 7", "100", "912 345623"]
        ]).toDF("NAME", "DOB", "ADDRESS", "SBP", "TEL")

 obfuscator = StructuredDeidentification(spark=spark, columns={"NAME": "ID", "DOB": "DATE"},
                                                      columnsSeed={"NAME": 23, "DOB": 23},
                                                      obfuscateRefSource="faker",
                                                      days=5
                                         )

result = obfuscator.obfuscateColumns(self.df)
result.show(truncate=False)                                             
```

*Results* :

```bash
+----------+------------+-----------------------+---+----------------+
|NAME      |DOB         |ADDRESS                |SBP|TEL             |
+----------+------------+-----------------------+---+----------------+
|[T1825511]|[18/02/1977]|711 Nulla St.          |140|673 431234      |
|[G6835267]|[28/02/1977]|1 Green Avenue.        |140|+23 (673) 431234|
|[S2371443]|[16/04/1900]|Calle del Libertador, 7|100|912 345623      |
+----------+------------+-----------------------+---+----------------+
```

#### New Multiple Chunks Merge Ability in ChunkMergeApproach

Updated ChunkMergeApproach to admit N input cols (`.setInputCols("ner_chunk","ner_chunk_1","ner_chunk_2")`). The input columns must be chunk columns.

*Example* :

```python
...
deid_ner = MedicalNerModel.pretrained("ner_deid_large", "en", "clinical/models") \
            .setInputCols(["sentence", "token", "embeddings"]) \
            .setOutputCol("ner")

ner_converter = NerConverter() \
            .setInputCols(["sentence", "token", "ner"]) \
            .setOutputCol("ner_chunk") \
            .setWhiteList(['DATE', 'AGE', 'NAME', 'PROFESSION', 'ID'])

medical_ner = MedicalNerModel.pretrained("ner_events_clinical", "en", "clinical/models") \
            .setInputCols(["sentence", "token", "embeddings"]) \
            .setOutputCol("ner2")

ner_converter_2 = NerConverter() \
            .setInputCols(["sentence", "token", "ner2"]) \
            .setOutputCol("ner_chunk_2")

ssn_parser = ContextualParserApproach() \
            .setInputCols(["sentence", "token"]) \
            .setOutputCol("entity_ssn") \
            .setJsonPath("../../src/test/resources/ssn.json") \
            .setCaseSensitive(False) \
            .setContextMatch(False)

chunk_merge = ChunkMergeApproach() \
            .setInputCols("entity_ssn","ner_chunk","ner_chunk_2") \
            .setOutputCol("deid_merged_chunk") \
            .setChunkPrecedence("field")      
...
```

#### New setBlackList Feature in ChunkMergeApproach

Now we can filter out the entities in the ChunkMergeApproach using a black list `.setBlackList(["NAME","ID"])`. The entities specified in the blackList will be excluded from the final entity list.

*Example* :

```python
chunk_merge = ChunkMergeApproach() \
            .setInputCols("entity_ssn","ner_chunk") \
            .setOutputCol("deid_merged_chunk") \
            .setBlackList(["NAME","ID"])
```

#### New setBlackList Feature in NerConverterInternal

Now we can filter out the entities in the NerConverterInternal using a black list `.setBlackList(["Drug","Treatment"])`. The entities specified in the blackList will be excluded from the final entity list.

*Example* :

```python
ner = MedicalNerModel.pretrained("ner_jsl_slim", "en", "clinical/models")\
        .setInputCols("sentence", "token","embeddings")\
        .setOutputCol("ner")

converter = NerConverterInternal()\
        .setInputCols("sentence","token","ner")\
        .setOutputCol("entities")\
        .setBlackList(["Drug","Treatment"])
```

#### New setLabelCasing Feature in MedicalNerModel

Now we can decide if we want to return the tags in upper or lower case with `setLabelCasing()`. That method convert the I-tags and B-tags in lower or upper case during the inference. The values will be 'lower' for lower case and 'upper' for upper case.


*Example* :

```python
...
ner_tagger = MedicalNerModel() \
            .pretrained("ner_clinical", "en", "clinical/models") \
            .setInputCols(["sentences", "tokens", "embeddings"]) \
            .setOutputCol("ner_tags") \
            .setLabelCasing("lower")
...

results = LightPipeline(pipelineModel).annotate("A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus ")
results["ner_tags"]
```

*Results* :

```bash
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-problem', 'I-problem', 'I-problem', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-problem', 'I-problem', 'I-problem', 'I-problem', 'I-problem']
```


#### New Update Models Functionality

We developed a new utility function called `UpdateModels` that allows you to refresh your `cache_pretrained` folder without running any annotator or manually checking. It has two methods;

+ `UpdateModels.updateCacheModels()` : This method lets you update all the models existing in the `cache_pretrained` folder. It downloads the latest version of all the models existing in the `cache_pretrained`.

*Example* :

```bash
# Models in /cache_pretrained
ls ~/cache_pretrained
>> ner_clinical_large_en_3.0.0_2.3_1617206114650/

# Update models in /cache_pretrained
from sparknlp_jsl.updateModels import UpdateModels
UpdateModels.updateCacheModels()
```

*Results* :

```bash
# Updated models in /cache_pretrained
ls ~/cache_pretrained
>> ner_clinical_large_en_3.0.0_2.3_1617206114650/
   ner_clinical_large_en_3.0.0_3.0_1617206114650/
```


+ `UpdateModels.updateModels("11/24/2021")` : This method lets you download all the new models uploaded to the Models Hub starting from a cut-off date (i.e. the last sync update).

*Example* :

```bash
# Models in /cache_pretrained
ls ~/cache_pretrained
>> ner_clinical_large_en_3.0.0_2.3_1617206114650/
   ner_clinical_large_en_3.0.0_3.0_1617206114650/

# Update models in /cache_pretrained according to date
from sparknlp_jsl.updateModels import UpdateModels
UpdateModels.updateModels("11/24/2021")

```

*Results* :

```bash
# Updated models in /cache_pretrained
ls ~/cache_pretrained
>>ner_clinical_large_en_3.0.0_2.3_1617206114650/
  ner_clinical_large_en_3.0.0_3.0_1617206114650/
  ner_model_finder_en_3.3.2_2.4_1637761259895/
  sbertresolve_ner_model_finder_en_3.3.2_2.4_1637764208798/
```


#### New and Updated Notebooks

+ We have a new [Connect to Annotation Lab via API Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Annotation_Lab/AL_API_import_export_pre_annotate.ipynb) you can find how to;

     - upload pre-annotations to ALAB
     - import a project form ALAB and convert to CoNLL file
     - upload tasks without pre-annotations

+ We have updated [Clinical Relation Extraction Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/10.Clinical_Relation_Extraction.ipynb) by adding a Relation Extraction Model-NER Model-Relation Pairs table that can be used to get the most optimal results when using these models.


**To see more, please check : [Spark NLP Healthcare Workshop Repo](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings/Healthcare)**

<div class="prev_ver h3-box" markdown="1">

## Versions

</div>

<ul class="pagination">
    <li>
        <a href="release_notes_3_3_2">Version 3.3.2</a>
    </li>
    <li>
        <strong>Version 3.3.4</strong>
    </li>
    <li>
        <a href="release_notes_3_4_0">Version 3.4.0</a>
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
    <li class="active"><a href="release_notes_3_3_4">3.3.4</a></li>
    <li><a href="release_notes_3_3_2">3.3.2</a></li>
    <li><a href="release_notes_3_3_1">3.3.1</a></li>
    <li><a href="release_notes_3_3_0">3.3.0</a></li>
    <li><a href="release_notes_3_2_3">3.2.3</a></li>
    <li><a href="release_notes_3_2_2">3.2.2</a></li>
    <li><a href="release_notes_3_2_1">3.2.1</a></li>
    <li><a href="release_notes_3_2_0">3.2.0</a></li>
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