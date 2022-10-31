---
layout: docs
header: true
seotitle: Spark NLP for Healthcare | John Snow Labs
title: Spark NLP for Healthcare Release Notes 3.4.0
permalink: /docs/en/spark_nlp_healthcare_versions/release_notes_3_4_0
key: docs-licensed-release-notes
modify_date: 2021-07-14
show_nav: true
sidebar:
    nav: sparknlp-healthcare
---

## 3.4.0

We are glad to announce that Spark NLP Healthcare 3.4.0 has been released!
This is a massive release: new features, new models, academic papers, and more!

#### Highlights

+ New German Deidentification NER Models
+ New German Deidentification Pretrained Pipeline
+ New Clinical NER Models
+ New AnnotationMerger Annotator
+ New MedicalBertForTokenClassifier Annotator
+ New BERT-Based Clinical NER Models
+ New Clinical Relation Extraction Models
+ New LOINC, SNOMED, UMLS and Clinical Abbreviation Entity Resolver Models
+ New ICD10 to ICD9 Code Mapping Pretrained Pipeline
+ New Clinical Sentence Embedding Models
+ Printing Validation and Test Logs for MedicalNerApproach and AssertionDLApproach
+ Filter Only the Regex Entities Feature in Deidentification Annotator
+ Add `.setMaskingPolicy` Parameter in Deidentification Annotator
+ Add `.cache_folder` Parameter in `UpdateModels.updateCacheModels()`
+ S3 Access Credentials No Longer Shipped Along Licenses
+ Enhanced Security for the Library and log4shell Update
+ New Peer-Reviewed Conference Paper on Clinical Relation Extraction
+ New Peer-Reviewed Conference Paper on Adverse Drug Events Extraction
+ New and Updated Notebooks

#### New German Deidentification NER Models

We trained two new NER models to find PHI data (protected health information) that may need to be deidentified in **German**.
`ner_deid_generic` and `ner_deid_subentity` models are trained with in-house annotations.

+ `ner_deid_generic` : Detects 7 PHI entities in German (`DATE`, `NAME`, `LOCATION`, `PROFESSION`, `CONTACT`, `AGE`, `ID`).

+ `ner_deid_subentity` : Detects 12 PHI sub-entities in German (`PATIENT`, `HOSPITAL`, `DATE`, `ORGANIZATION`, `CITY`, `STREET`, `USERNAME`, `PROFESSION`, `PHONE`, `COUNTRY`, `DOCTOR`, `AGE`).

*Example* :

```bash
...

embeddings = WordEmbeddingsModel.pretrained("w2v_cc_300d","de","clinical/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")

deid_ner = MedicalNerModel.pretrained("ner_deid_generic", "de", "clinical/models")\
  .setInputCols(["sentence", "token", "embeddings"])\
  .setOutputCol("ner")

deid_sub_entity_ner = MedicalNerModel.pretrained("ner_deid_subentity", "de", "clinical/models")\
  .setInputCols(["sentence", "token", "embeddings"])\
  .setOutputCol("ner_sub_entity")
...

text = """Michael Berger wird am Morgen des 12 Dezember 2018 ins St. Elisabeth-Krankenhaus
in Bad Kissingen eingeliefert. Herr Berger ist 76 Jahre alt und hat zu viel Wasser in den Beinen."""

result = model.transform(spark.createDataFrame([[text]], ["text"]))
```

*Results* :

```bash
+-------------------------+----------------------+-------------------------+
|chunk                    |ner_deid_generic_chunk|ner_deid_subentity_chunk |
+-------------------------+----------------------+-------------------------+
|Michael Berger           |NAME                  |PATIENT                  |
|12 Dezember 2018         |DATE                  |DATE                     |
|St. Elisabeth-Krankenhaus|LOCATION              |HOSPITAL                 |
|Bad Kissingen            |LOCATION              |CITY                     |
|Berger                   |NAME                  |PATIENT                  |
|76                       |AGE                   |AGE                      |
+-------------------------+----------------------+-------------------------+
```
#### New German Deidentification Pretrained Pipeline

We developed a clinical deidentification pretrained pipeline that can be used to deidentify PHI information from **German** medical texts. The PHI information will be masked and obfuscated in the resulting text. The pipeline can mask and obfuscate `PATIENT`, `HOSPITAL`, `DATE`, `ORGANIZATION`, `CITY`, `STREET`, `USERNAME`, `PROFESSION`, `PHONE`, `COUNTRY`, `DOCTOR`, `AGE`, `CONTACT`, `ID`, `PHONE`, `ZIP`, `ACCOUNT`, `SSN`, `DLN`, `PLATE` entities.

*Example* :

```bash
...
from sparknlp.pretrained import PretrainedPipeline

deid_pipeline = PretrainedPipeline("clinical_deidentification", "de", "clinical/models")

text = """Zusammenfassung : Michael Berger wird am Morgen des 12 Dezember 2018 ins St.Elisabeth Krankenhaus in Bad Kissingen eingeliefert.
Herr Michael Berger ist 76 Jahre alt und hat zu viel Wasser in den Beinen.

Persönliche Daten :
ID-Nummer: T0110053F
Platte A-BC124
Kontonummer: DE89370400440532013000
SSN : 13110587M565
Lizenznummer: B072RRE2I55
Adresse : St.Johann-Straße 13 19300"""

result = deid_pipe.annotate(text)

print("\n".join(result['masked']))
print("\n".join(result['obfuscated']))
print("\n".join(result['masked_with_chars']))
print("\n".join(result['masked_fixed_length_chars']))

```
*Results* :

```bash
Zusammenfassung : <PATIENT> wird am Morgen des <DATE> ins <HOSPITAL> eingeliefert.
Herr <PATIENT> ist <AGE> Jahre alt und hat zu viel Wasser in den Beinen.
Persönliche Daten :
ID-Nummer: <ID>
Platte <PLATE>
Kontonummer: <ACCOUNT>
SSN : <SSN>
Lizenznummer: <DLN>
Adresse : <STREET> <ZIP>

Zusammenfassung : Herrmann Kallert wird am Morgen des 11-26-1977 ins International Neuroscience eingeliefert.
Herr Herrmann Kallert ist 79 Jahre alt und hat zu viel Wasser in den Beinen.
Persönliche Daten :
ID-Nummer: 136704D357
Platte QA348G
Kontonummer: 192837465738
SSN : 1310011981M454
Lizenznummer: XX123456
Adresse : Klingelhöferring 31206

Zusammenfassung : **** wird am Morgen des **** ins **** eingeliefert.
Herr **** ist **** Jahre alt und hat zu viel Wasser in den Beinen.
Persönliche Daten :
ID-Nummer: ****
Platte ****
Kontonummer: ****
SSN : ****
Lizenznummer: ****
Adresse : **** ****

Zusammenfassung : [************] wird am Morgen des [**************] ins [**********************] eingeliefert.
Herr [************] ist ** Jahre alt und hat zu viel Wasser in den Beinen.
Persönliche Daten :
ID-Nummer: [*******]
Platte [*****]
Kontonummer: [********************]
SSN : [**********]
Lizenznummer: [*********]
Adresse : [*****************] [***]
```

#### New Clinical NER Models

We have two new clinical NER models.

+ `ner_abbreviation_clinical` : This model is trained to extract clinical abbreviations and acronyms in texts and labels these entities as `ABBR`.

*Example* :

```bash
...
clinical_ner = MedicalNerModel.pretrained("ner_abbreviation_clinical", "en", "clinical/models")\
  .setInputCols(["sentence", "token", "embeddings"])\
  .setOutputCol("ner")
...

results = ner_model.transform(spark.createDataFrame([["Gravid with estimated fetal weight of 6-6/12 pounds. LOWER EXTREMITIES: No edema. LABORATORY DATA: Laboratory tests include a CBC which is normal. Blood Type: AB positive. Rubella: Immune. VDRL: Nonreactive. Hepatitis C surface antigen: Negative. HIV: Negative. One-Hour Glucose: 117. Group B strep has not been done as yet."]], ["text"]))
```

*Results* :

```bash
+-----+---------+
|chunk|ner_label|
+-----+---------+
|CBC  |ABBR     |
|AB   |ABBR     |
|VDRL |ABBR     |
|HIV  |ABBR     |
+-----+---------+
```

+ `ner_drugprot_clinical` : This model detects chemical compounds/drugs and genes/proteins in medical text and research articles. Here are the labels it can detect : `GENE`, `CHEMICAL`, `GENE_AND_CHEMICAL`.

*Example* :

```bash
...
clinical_ner = MedicalNerModel.pretrained("ner_drugprot_clinical", "en", "clinical/models")\
  .setInputCols(["sentence", "token", "embeddings"])\
  .setOutputCol("ner")
...

results = ner_model.transform(spark.createDataFrame([["Anabolic effects of clenbuterol on skeletal muscle are mediated by beta 2-adrenoceptor activation"]], ["text"]))
```

*Results* :

```bash
|    | chunk                | ner_label         |
|---:|:---------------------|:------------------|
|  0 | clenbuterol          | CHEMICAL          |
|  1 | beta 2-adrenoceptor  | GENE              |

```

#### New AnnotationMerger Annotator

A new annotator: `AnnotationMerger`. Besides NERs, now we will be able to merge results of **Relation Extraction models** and **Assertion models** as well. Therefore, it can merge results of Relation Extraction models, NER models, and Assertion Status models.

*Example-1* :

```bash
...
annotation_merger = AnnotationMerger()\
    .setInputCols("ade_relations", "pos_relations", "events_relations")\
    .setInputType("category")\
    .setOutputCol("all_relations")
...

results = ann_merger_model.transform(spark.createDataFrame([["The patient was prescribed 1 unit of naproxen for 5 days after meals for chronic low back pain. The patient was also given 1 unit of oxaprozin daily for rheumatoid arthritis presented with tense bullae and cutaneous fragility on the face and the back of the hands."]], ["text"]))
```

*Results-1* :

```bash
|    | all_relations   | all_relations_entity1   | all_relations_chunk1   | all_relations_entity2   | all_relations_chunk2                                      |
|---:|:----------------|:------------------------|:-----------------------|:------------------------|:----------------------------------------------------------|
|  0 | 1               | DRUG                    | oxaprozin              | ADE                     | tense bullae                                              |
|  1 | 1               | DRUG                    | oxaprozin              | ADE                     | cutaneous fragility on the face and the back of the hands |
|  2 | DOSAGE-DRUG     | DOSAGE                  | 1 unit                 | DRUG                    | naproxen                                                  |
|  3 | DRUG-DURATION   | DRUG                    | naproxen               | DURATION                | for 5 days                                                |
|  4 | DOSAGE-DRUG     | DOSAGE                  | 1 unit                 | DRUG                    | oxaprozin                                                 |
|  5 | DRUG-FREQUENCY  | DRUG                    | oxaprozin              | FREQUENCY               | daily                                                     |
|  6 | OVERLAP         | TREATMENT               | naproxen               | DURATION                | 5 days                                                    |
|  7 | OVERLAP         | TREATMENT               | oxaprozin              | FREQUENCY               | daily                                                     |
|  8 | BEFORE          | TREATMENT               | oxaprozin              | PROBLEM                 | rheumatoid arthritis                                      |
|  9 | AFTER           | TREATMENT               | oxaprozin              | OCCURRENCE              | presented                                                 |
| 10 | OVERLAP         | FREQUENCY               | daily                  | PROBLEM                 | rheumatoid arthritis                                      |
| 11 | OVERLAP         | FREQUENCY               | daily                  | PROBLEM                 | tense bullae                                              |
| 12 | OVERLAP         | FREQUENCY               | daily                  | PROBLEM                 | cutaneous fragility on the face                           |
| 13 | BEFORE          | PROBLEM                 | rheumatoid arthritis   | OCCURRENCE              | presented                                                 |
| 14 | OVERLAP         | PROBLEM                 | rheumatoid arthritis   | PROBLEM                 | tense bullae                                              |
| 15 | OVERLAP         | PROBLEM                 | rheumatoid arthritis   | PROBLEM                 | cutaneous fragility on the face                           |
| 16 | BEFORE          | OCCURRENCE              | presented              | PROBLEM                 | tense bullae                                              |
| 17 | BEFORE          | OCCURRENCE              | presented              | PROBLEM                 | cutaneous fragility on the face                           |
| 18 | OVERLAP         | PROBLEM                 | tense bullae           | PROBLEM                 | cutaneous fragility on the face                           |
```

*Example-2* :

```bash
...
ner_annotation_merger = AnnotationMerger()\
    .setInputCols("ner_chunk", "radiology_ner_chunk", "jsl_ner_chunk")\
    .setInputType("chunk")\
    .setOutputCol("all_ners")

assertion_annotation_merger = AnnotationMerger()\
    .setInputCols("clinical_assertion", "radiology_assertion", "jsl_assertion")\
    .setInputType("assertion")\
    .setOutputCol("all_assertions")
...

results = ann_merger_model.transform(spark.createDataFrame([["The patient was prescribed 1 unit of naproxen for 5 days after meals for chronic low back pain. The patient was also given 1 unit of oxaprozin daily for rheumatoid arthritis presented with tense bullae and cutaneous fragility on the face and the back of the hands."]], ["text"]))
```

*Results-2* :

```bash
|    | ners                            | all_assertions   |
|---:|:--------------------------------|:-----------------|
|  0 | naproxen                        | present          |
|  1 | chronic low back pain           | present          |
|  2 | oxaprozin                       | present          |
|  3 | rheumatoid arthritis            | present          |
|  4 | tense bullae                    | present          |
|  5 | cutaneous fragility on the face | present          |
|  6 | low back                        | Confirmed        |
|  7 | pain                            | Confirmed        |
|  8 | rheumatoid arthritis            | Confirmed        |
|  9 | tense bullae                    | Confirmed        |
| 10 | cutaneous                       | Confirmed        |
| 11 | fragility                       | Confirmed        |
| 12 | face                            | Confirmed        |
| 13 | back                            | Confirmed        |
| 14 | hands                           | Confirmed        |
| 15 | 1 unit                          | Present          |
| 16 | naproxen                        | Past             |
| 17 | for 5 days                      | Past             |
| 18 | chronic                         | Someoneelse      |
| 19 | low                             | Past             |
| 20 | back pain                       | Present          |
| 21 | 1 unit                          | Past             |
| 22 | oxaprozin                       | Past             |
| 23 | daily                           | Past             |
| 24 | rheumatoid arthritis            | Present          |
| 25 | tense                           | Present          |
| 26 | bullae                          | Present          |
| 27 | cutaneous fragility             | Present          |
| 28 | face                            | Someoneelse      |
| 29 | back of the hands               | Present          |
```

#### New MedicalBertForTokenClassifier Annotator

We developed a new annotator called MedicalBertForTokenClassifier that can load BERT-Based clinical token classifier models head on top (a linear layer on top of the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks.


#### New BERT-Based Clinical NER Models

Here are the MedicalBertForTokenClassifier Models we have in the library at the moment:

+ `bert_token_classifier_ner_ade`
+ `bert_token_classifier_ner_anatomy`
+ `bert_token_classifier_ner_bionlp`
+ `bert_token_classifier_ner_cellular`
+ `bert_token_classifier_ner_chemprot`
+ `bert_token_classifier_ner_chemicals`
+ `bert_token_classifier_ner_jsl_slim`
+ `bert_token_classifier_ner_jsl`
+ `bert_token_classifier_ner_deid`
+ `bert_token_classifier_ner_drugs`
+ `bert_token_classifier_ner_clinical`
+ `bert_token_classifier_ner_bacteria`

In addition, we are releasing a new BERT-Based clinical NER model named `bert_token_classifier_drug_development_trials`. It is a `MedicalBertForTokenClassification` NER model to identify concepts related to drug development including `Trial Groups` , `End Points` , `Hazard Ratio`, and other entities in free text. It can detect the following entities: `Patient_Count`, `Duration`, `End_Point`, `Value`, `Trial_Group`, `Hazard_Ratio`, `Total_Patients`

*Example* :

```bash
...
tokenClassifier= MedicalBertForTokenClassifier.pretrained("bert_token_classifier_drug_development_trials", "en", "clinical/models")\
  .setInputCols("token", "document")\
  .setOutputCol("ner")\
  .setCaseSensitive(True)
...

results = ner_model.transform(spark.createDataFrame([["In June 2003, the median overall survival with and without topotecan were 4.0 and 3.6 months, respectively. The best complete response ( CR ) , partial response ( PR ) , stable disease and progressive disease were observed in 23, 63, 55 and 33 patients, respectively, with topotecan, and 11, 61, 66 and 32 patients, respectively, without topotecan."]], ["text"]))

```

*Results* :

```bash
|    | chunk             | entity        |
|---:|:------------------|:--------------|
|  0 | median            | Duration      |
|  1 | overall survival  | End_Point     |
|  2 | with              | Trial_Group   |
|  3 | without topotecan | Trial_Group   |
|  4 | 4.0               | Value         |
|  5 | 3.6 months        | Value         |
|  6 | 23                | Patient_Count |
|  7 | 63                | Patient_Count |
|  8 | 55                | Patient_Count |
|  9 | 33 patients       | Patient_Count |
| 10 | topotecan         | Trial_Group   |
| 11 | 11                | Patient_Count |
| 12 | 61                | Patient_Count |
| 13 | 66                | Patient_Count |
| 14 | 32 patients       | Patient_Count |
| 15 | without topotecan | Trial_Group   |
```

#### New Clinical Relation Extraction Models

We have two new clinical Relation Extraction models for detecting interactions between drugs and proteins. These models work hand-in-hand with the new `ner_drugprot_clinical` NER model and detect following relations between entities: `INHIBITOR`, `DIRECT-REGULATOR`, `SUBSTRATE`, `ACTIVATOR`, `INDIRECT-UPREGULATOR`, `INDIRECT-DOWNREGULATOR`, `ANTAGONIST`, `PRODUCT-OF`, `PART-OF`, `AGONIST`.

+ `redl_drugprot_biobert` : This model was trained using BERT and performs with higher accuracy.

+ `re_drugprot_clinical` : This model was trained using `RelationExtractionApproach()`.

*Example* :

```bash
...
drugprot_ner_tagger = MedicalNerModel.pretrained("ner_drugprot_clinical", "en", "clinical/models")\
    .setInputCols("sentences", "tokens", "embeddings")\
    .setOutputCol("ner_tags")   
...

drugprot_re_biobert = RelationExtractionDLModel()\
    .pretrained('redl_drugprot_biobert', "en", "clinical/models")\
    .setPredictionThreshold(0.9)\
    .setInputCols(["re_ner_chunks", "sentences"])\
    .setOutputCol("relations")

drugprot_re_clinical = RelationExtractionModel()\
    .pretrained("re_drugprot_clinical", "en", 'clinical/models')\
    .setInputCols(["embeddings", "pos_tags", "ner_chunks", "dependencies"])\
    .setOutputCol("relations")\
    .setMaxSyntacticDistance(4)\
    .setPredictionThreshold(0.9)\
    .setRelationPairs(['CHEMICAL-GENE'])
...

sample_text = "Lipid specific activation of the murine P4-ATPase Atp8a1 (ATPase II). The asymmetric transbilayer distribution of phosphatidylserine (PS) in the mammalian plasma membrane and secretory vesicles is maintained, in part, by an ATP-dependent transporter. This aminophospholipid "flippase" selectively transports PS to the cytosolic leaflet of the bilayer and is sensitive to vanadate, Ca(2+), and modification by sulfhydryl reagents. Although the flippase has not been positively identified, a subfamily of P-type ATPases has been proposed to function as transporters of amphipaths, including PS and other phospholipids. A candidate PS flippase ATP8A1 (ATPase II), originally isolated from bovine secretory vesicles, is a member of this subfamily based on sequence homology to the founding member of the subfamily, the yeast protein Drs2, which has been linked to ribosomal assembly, the formation of Golgi-coated vesicles, and the maintenance of PS asymmetry."
result = re_model.transform(spark.createDataFrame([[sample_text]]).toDF("text"))
```

*Results* :

```bash
+---------+--------+-------------+-----------+--------------------+-------+-------------+-----------+--------------------+----------+
| relation| entity1|entity1_begin|entity1_end|              chunk1|entity2|entity2_begin|entity2_end|              chunk2|confidence|
+---------+--------+-------------+-----------+--------------------+-------+-------------+-----------+--------------------+----------+
|SUBSTRATE|CHEMICAL|          308|        310|                  PS|   GENE|          275|        283|            flippase|  0.998399|
|ACTIVATOR|CHEMICAL|         1563|       1578|     sn-1,2-glycerol|   GENE|         1479|       1509|plasma membrane P...|  0.999304|
|ACTIVATOR|CHEMICAL|         1563|       1578|     sn-1,2-glycerol|   GENE|         1511|       1517|              Atp8a1|  0.979057|
+---------+--------+-------------+-----------+--------------------+-------+-------------+-----------+--------------------+----------+
```

#### New LOINC, SNOMED, UMLS and Clinical Abbreviation Entity Resolver Models

We have five new Sentence Entity Resolver models.

+ `sbiobertresolve_clinical_abbreviation_acronym` : This model maps clinical abbreviations and acronyms to their meanings using `sbiobert_base_cased_mli` Sentence Bert Embeddings. It is a part of ongoing research we have been running in-house, and trained with a limited dataset. We’ll be updating & enriching the model in the upcoming releases.

*Example* :

```bash
...
abbr_resolver = SentenceEntityResolverModel.pretraind("sbiobertresolve_clinical_abbreviation_acronym", "en", "clinical/models")\
  .setInputCols(["merged_chunk", "sentence_embeddings"])\
  .setOutputCol("abbr_meaning")\
  .setDistanceFunction("EUCLIDEAN")
...

sample_text = "HISTORY OF PRESENT ILLNESS: The patient three weeks ago was seen at another clinic for upper respiratory infection-type symptoms. She was diagnosed with a viral infection and had used OTC medications including Tylenol, Sudafed, and Nyquil."
results = abb_model.transform(spark.createDataFrame([[sample_text]]).toDF('text'))
```

*Results* :

```bash
|   sent_id | ner_chunk   | entity   | abbr_meaning     | all_k_results                                                                      | all_k_resolutions          |
|----------:|:------------|:---------|:-----------------|:-----------------------------------------------------------------------------------|:---------------------------|
|         0 | OTC         | ABBR     | over the counter | ['over the counter', 'ornithine transcarbamoylase', 'enteric-coated', 'thyroxine'] | ['OTC', 'OTC', 'EC', 'T4'] |

```

+ `sbiobertresolve_umls_drug_substance` : This model maps clinical entities to UMLS CUI codes. It is trained on `2021AB` UMLS dataset. The complete dataset has 127 different categories, and this model is trained on the `Clinical Drug`, `Pharmacologic Substance`, `Antibiotic`, `Hazardous or Poisonous Substance` categories using `sbiobert_base_cased_mli` embeddings.

*Example* :

```bash
...
umls_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_umls_drug_substance","en", "clinical/models")\
  .setInputCols(["ner_chunk", "sbert_embeddings"])\
  .setOutputCol("resolution")\
  .setDistanceFunction("EUCLIDEAN")
...

results = model.fullAnnotate(['Dilaudid', 'Hydromorphone', 'Exalgo', 'Palladone', 'Hydrogen peroxide 30 mg', 'Neosporin Cream', 'Magnesium hydroxide 100mg/1ml', 'Metformin 1000 mg'])
```

*Results* :

```bash
|    | chunk                         | code     | code_description           | all_k_code_desc                                              | all_k_codes                                                                                                                                                                             |
|---:|:------------------------------|:---------|:---------------------------|:-------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|  0 | Dilaudid                      | C0728755 | dilaudid                   | ['C0728755', 'C0719907', 'C1448344', 'C0305924', 'C1569295'] | ['dilaudid', 'Dilaudid HP', 'Disthelm', 'Dilaudid Injection', 'Distaph']                                                                                                                |
|  1 | Hydromorphone                 | C0012306 | HYDROMORPHONE              | ['C0012306', 'C0700533', 'C1646274', 'C1170495', 'C0498841'] | ['HYDROMORPHONE', 'Hydromorphone HCl', 'Phl-HYDROmorphone', 'PMS HYDROmorphone', 'Hydromorphone injection']                                                                             |
|  2 | Exalgo                        | C2746500 | Exalgo                     | ['C2746500', 'C0604734', 'C1707065', 'C0070591', 'C3660437'] | ['Exalgo', 'exaltolide', 'Exelgyn', 'Extacol', 'exserohilone']                                                                                                                          |
|  3 | Palladone                     | C0730726 | palladone                  | ['C0730726', 'C0594402', 'C1655349', 'C0069952', 'C2742475'] | ['palladone', 'Palladone-SR', 'Palladone IR', 'palladiazo', 'palladia']                                                                                                                 |
|  4 | Hydrogen peroxide 30 mg       | C1126248 | hydrogen peroxide 30 MG/ML | ['C1126248', 'C0304655', 'C1605252', 'C0304656', 'C1154260'] | ['hydrogen peroxide 30 MG/ML', 'Hydrogen peroxide solution 30%', 'hydrogen peroxide 30 MG/ML [Proxacol]', 'Hydrogen peroxide 30 mg/mL cutaneous solution', 'benzoyl peroxide 30 MG/ML'] |
|  5 | Neosporin Cream               | C0132149 | Neosporin Cream            | ['C0132149', 'C0306959', 'C4722788', 'C0704071', 'C0698988'] | ['Neosporin Cream', 'Neosporin Ointment', 'Neomycin Sulfate Cream', 'Neosporin Topical Ointment', 'Naseptin cream']                                                                     |
|  6 | Magnesium hydroxide 100mg/1ml | C1134402 | magnesium hydroxide 100 MG | ['C1134402', 'C1126785', 'C4317023', 'C4051486', 'C4047137'] | ['magnesium hydroxide 100 MG', 'magnesium hydroxide 100 MG/ML', 'Magnesium sulphate 100mg/mL injection', 'magnesium sulfate 100 MG', 'magnesium sulfate 100 MG/ML']                     |
|  7 | Metformin 1000 mg             | C0987664 | metformin 1000 MG          | ['C0987664', 'C2719784', 'C0978482', 'C2719786', 'C4282269'] | ['metformin 1000 MG', 'metFORMIN hydrochloride 1000 MG', 'METFORMIN HCL 1000MG TAB', 'metFORMIN hydrochloride 1000 MG [Fortamet]', 'METFORMIN HCL 1000MG SA TAB']                       |

```

+ `sbiobertresolve_loinc_cased` : This model maps extracted clinical NER entities to LOINC codes using `sbiobert_base_cased_mli` Sentence Bert Embeddings. It is trained with augmented **cased** concept names since sbiobert model is cased.

*Example* :

```bash
...
loinc_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_loinc_cased", "en", "clinical/models")\
  .setInputCols(["ner_chunk", "sbert_embeddings"])\
  .setOutputCol("resolution")\
  .setDistanceFunction("EUCLIDEAN")
...

sample_text= """The patient is a 22-year-old female with a history of obesity. She has a BMI of 33.5 kg/m2, aspartate aminotransferase 64, and alanine aminotransferase 126. Her hemoglobin is 8.2%."""
result = model.transform(spark.createDataFrame([[sample_text]], ["text"]))
```

*Results* :

```bash
+-------------------------------------+------+-----------+----------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|                            ner_chunk|entity| resolution|                                           all_codes|                                                                                                                                                                                             resolutions|
+-------------------------------------+------+-----------+----------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|                                  BMI|  Test|  LP35925-4|[LP35925-4, 59574-4, BDYCRC, 73964-9, 59574-4,...   |[Body mass index (BMI), Body mass index, Body circumference, Body muscle mass, Body mass index (BMI) [Percentile], ...                                                                                  |
|           aspartate aminotransferase|  Test|    14409-7|[14409-7, 1916-6, 16325-3, 16324-6, 43822-6, 308... |[Aspartate aminotransferase, Aspartate aminotransferase/Alanine aminotransferase, Alanine aminotransferase/Aspartate aminotransferase, Alanine aminotransferase, Aspartate aminotransferase [Prese...   |
|             alanine aminotransferase|  Test|    16324-6|[16324-6, 16325-3, 14409-7, 1916-6, 59245-1, 30...  |[Alanine aminotransferase, Alanine aminotransferase/Aspartate aminotransferase, Aspartate aminotransferase, Aspartate aminotransferase/Alanine aminotransferase, Alanine glyoxylate aminotransfer,...   |
|                           hemoglobin|  Test|    14775-1|[14775-1, 16931-8, 12710-0, 29220-1, 15082-1, 72... |[Hemoglobin, Hematocrit/Hemoglobin, Hemoglobin pattern, Haptoglobin, Methemoglobin, Oxyhemoglobin, Hemoglobin test status, Verdohemoglobin, Hemoglobin A, Hemoglobin distribution width, Myoglobin,...  |
+-------------------------------------+------+-----------+----------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
```

+ `sbluebertresolve_loinc_uncased` : This model maps extracted clinical NER entities to LOINC codes using `sbluebert_base_uncased_mli` Sentence Bert Embeddings. It trained on the augmented version of the **uncased (lowercased)** dataset which is used in previous LOINC resolver models.

*Example* :

```bash
...
loinc_resolver = SentenceEntityResolverModel.pretrained("sbluebertresolve_loinc_uncased", "en", "clinical/models")\
  .setInputCols(["jsl_ner_chunk", "sbert_embeddings"])\
  .setOutputCol("resolution")\
  .setDistanceFunction("EUCLIDEAN")
...

sample_text= """The patient is a 22-year-old female with a history of obesity. She has a BMI of 33.5 kg/m2, aspartate aminotransferase 64, and alanine aminotransferase 126. Her hgba1c is 8.2%."""
result = model.transform(spark.createDataFrame([[sample_text]], ["text"]))
```
*Results* :

```bash
+-------------------------------------+------+-----------+----------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|                            ner_chunk|entity| resolution|                                           all_codes|                                                                                                                                                                                             resolutions|
+-------------------------------------+------+-----------+----------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|                                  BMI|  Test|    39156-5|[39156-5, LP35925-4, BDYCRC, 73964-9, 59574-4,...]  |[Body mass index, Body mass index (BMI), Body circumference, Body muscle mass, Body mass index (BMI) [Percentile], ...]                                                                                 |
|           aspartate aminotransferase|  Test|    14409-7|['14409-7', '16325-3', '1916-6', '16324-6',...]     |['Aspartate aminotransferase', 'Alanine aminotransferase/Aspartate aminotransferase', 'Aspartate aminotransferase/Alanine aminotransferase', 'Alanine aminotransferase', ...]                           |
|             alanine aminotransferase|  Test|    16324-6|['16324-6', '1916-6', '16325-3', '59245-1',...]     |['Alanine aminotransferase', 'Aspartate aminotransferase/Alanine aminotransferase', 'Alanine aminotransferase/Aspartate aminotransferase', 'Alanine glyoxylate aminotransferase',...]                   |
|                               hgba1c|  Test|    41995-2|['41995-2', 'LP35944-5', 'LP19717-5', '43150-2',...]|['Hemoglobin A1c', 'HbA1c measurement device', 'HBA1 gene', 'HbA1c measurement device panel', ...]                                                                                                      |
+-------------------------------------+------+-----------+------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
```

+ `sbiobertresolve_snomed_drug` : This model maps detected drug entities to SNOMED codes using `sbiobert_base_cased_mli` Sentence Bert Embeddings.

*Example* :

```bash
...
snomed_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_snomed_drug", "en", "clinical/models") \
    .setInputCols(["ner_chunk", "sentence_embeddings"]) \
    .setOutputCol("snomed_code")\
    .setDistanceFunction("EUCLIDEAN")
...

sample_text = "She is given Fragmin 5000 units subcutaneously daily, OxyContin 30 mg p.o. q.12 h., folic acid 1 mg daily, levothyroxine 0.1 mg p.o. daily, Avandia 4 mg daily, aspirin 81 mg daily, Neurontin 400 mg p.o. t.i.d., magnesium citrate 1 bottle p.o. p.r.n., sliding scale coverage insulin."
results = model.transform(spark.createDataFrame([[sample_text]]).toDF('text'))
```

*Results* :

```bash
+-----------------+------+-----------------+-----------------+------------------------------------------------------------+------------------------------------------------------------+
|        ner_chunk|entity|      snomed_code|    resolved_text|                                               all_k_results|                                           all_k_resolutions|
+-----------------+------+-----------------+-----------------+------------------------------------------------------------+------------------------------------------------------------+
|          Fragmin|  DRUG| 9487801000001106|          Fragmin|9487801000001106:::130752006:::28999000:::953500100000110...|Fragmin:::Fragilysin:::Fusarin:::Femulen:::Fumonisin:::Fr...|
|        OxyContin|  DRUG| 9296001000001100|        OxyCONTIN|9296001000001100:::373470001:::230091000001108:::55452001...|OxyCONTIN:::Oxychlorosene:::Oxyargin:::oxyCODONE:::Oxymor...|
|       folic acid|  DRUG|         63718003|       Folic acid|63718003:::6247001:::226316008:::432165000:::438451000124...|Folic acid:::Folic acid-containing product:::Folic acid s...|
|    levothyroxine|  DRUG|10071011000001106|    Levothyroxine|10071011000001106:::710809001:::768532006:::126202002:::7...|Levothyroxine:::Levothyroxine (substance):::Levothyroxine...|
|          Avandia|  DRUG| 9217601000001109|          avandia|9217601000001109:::9217501000001105:::12226401000001108::...|avandia:::avandamet:::Anatera:::Intanza:::Avamys:::Aragam...|
|          aspirin|  DRUG|        387458008|          Aspirin|387458008:::7947003:::5145711000001107:::426365001:::4125...|Aspirin:::Aspirin-containing product:::Aspirin powder:::A...|
|        Neurontin|  DRUG| 9461401000001102|        neurontin|9461401000001102:::130694004:::86822004:::952840100000110...|neurontin:::Neurolysin:::Neurine (substance):::Nebilet:::...|
|magnesium citrate|  DRUG|         12495006|Magnesium citrate|12495006:::387401007:::21691008:::15531411000001106:::408...|Magnesium citrate:::Magnesium carbonate:::Magnesium trisi...|
|          insulin|  DRUG|         67866001|          Insulin|67866001:::325072002:::414515005:::39487003:::411530000::...|Insulin:::Insulin aspart:::Insulin detemir:::Insulin-cont...|
+-----------------+------+-----------------+-----------------+------------------------------------------------------------+------------------------------------------------------------+

```

#### New ICD10 to ICD9 Code Mapping Pretrained Pipeline

We are releasing new `icd10_icd9_mapping` pretrained pipeline. This pretrained pipeline maps ICD10 codes to ICD9 codes without using any text data. You’ll just feed a comma or white space-delimited ICD10 codes and it will return the corresponding ICD9 codes as a list.

*Example* :

```bash
from sparknlp.pretrained import PretrainedPipeline
pipeline = PretrainedPipeline("icd10_icd9_mapping", "en", "clinical/models")
pipeline.annotate('E669 R630 J988')
```
*Results* :

```bash
{'document': ['E669 R630 J988'],
'icd10': ['E669', 'R630', 'J988'],
'icd9': ['27800', '7830', '5198']}

Code Descriptions:

|    | ICD10                | Details                               |
|---:|:---------------------|:--------------------------------------|
|  0 | E669                 | Obesity                               |
|  1 | R630                 | Anorexia                              |
|  2 | J988                 | Other specified respiratory disorders |

|    | ICD9                 | Details                               |
|---:|:---------------------|:--------------------------------------|
|  0 | 27800                | Obesity                               |
|  1 | 7830                 | Anorexia                              |
|  2 | 5198                 | Other diseases of respiratory system  |

```

#### New Clinical Sentence Embedding Models

We have two new clinical Sentence Embedding models.

+ `sbiobert_jsl_rxnorm_cased` : This model maps sentences & documents to a 768 dimensional dense vector space by using average pooling on top of BioBert model. It's also fine-tuned on RxNorm dataset to help generalization over medication-related datasets.

*Example* :

```bash
...
sentence_embeddings = BertSentenceEmbeddings.pretrained("sbiobert_jsl_rxnorm_cased", "en", "clinical/models")\
  .setInputCols(["sentence"])\
  .setOutputCol("sbioert_embeddings")
...
```

+ `sbert_jsl_medium_rxnorm_uncased` : This model maps sentences & documents to a 512-dimensional dense vector space by using average pooling on top of BERT model. It's also fine-tuned on the RxNorm dataset to help generalization over medication-related datasets.

*Example* :

```bash
...
sentence_embeddings = BertSentenceEmbeddings.pretrained("sbert_jsl_medium_rxnorm_uncased", "en", "clinical/models")\
  .setInputCols(["sentence"])\
  .setOutputCol("sbert_embeddings")
...
```

#### Printing Validation and Test Logs in MedicalNerApproach and AssertionDLApproach

Now we can check validation loss and test loss for each epoch in the logs created during trainings of MedicalNerApproach and AssertionDLApproach.


```bash
Epoch 15/15 started, lr: 9.345794E-4, dataset size: 1330


Epoch 15/15 - 56.65s - loss: 37.58828 - avg training loss: 1.7899181 - batches: 21
Quality on validation dataset (20.0%), validation examples = 266
time to finish evaluation: 8.11s
Total validation loss: 15.1930	Avg validation loss: 2.5322
label	 tp	 fp	 fn	 prec	 rec	 f1
I-Disease	 707	 72	 121	 0.9075738	 0.8538647	 0.8799004
B-Disease	 657	 81	 60	 0.8902439	 0.916318	 0.90309274
tp: 1364 fp: 153 fn: 181 labels: 2
Macro-average	 prec: 0.89890885, rec: 0.88509136, f1: 0.8919466
Micro-average	 prec: 0.89914304, rec: 0.8828479, f1: 0.89092094
Quality on test dataset:
time to finish evaluation: 9.11s
Total test loss: 17.7705	Avg test loss: 1.6155
label	 tp	 fp	 fn	 prec	 rec	 f1
I-Disease	 663	 113	 126	 0.85438144	 0.8403042	 0.8472843
B-Disease	 631	 122	 77	 0.8379814	 0.8912429	 0.86379194
tp: 1294 fp: 235 fn: 203 labels: 2
Macro-average	 prec: 0.8461814, rec: 0.86577356, f1: 0.85586536
Micro-average	 prec: 0.8463048, rec: 0.86439544, f1: 0.8552544
```

#### Filter Only the Regex Entities Feature in Deidentification Annotator

The `setBlackList()` method will be able to filter just the detected Regex Entities. Before this change we filtered the chunks and the regex entities.

#### Add `.setMaskingPolicy` Parameter in Deidentification Annotator

Now we can have three modes to mask the entities in the Deidentification annotator.
You can select the modes using the `.setMaskingPolicy("entity_labels")`.

The methods are the followings:
  1. "entity_labels": Mask with the entity type of that chunk. (default)
  2. "same_length_chars": Mask the deid entities with same length of asterix (`*`) with brackets (`[`,`]`) on both end.
  3. "fixed_length_chars": Mask the deid entities with a fixed length of asterix (`*`). The length is setting up using the `setFixedMaskLength(4)` method.


Given the following sentence `John Snow is a good guy.` the result will be:

  1. "entity_labels": `<NAME> is a good guy.`
  2. "same_length_chars": `[*******] is a good guy.`
  3. "fixed_length_chars": `**** is a good guy.`

*Example*
```bash
Masked with entity labels
------------------------------
DATE <DATE>, <DOCTOR>,  The driver's license <DLN>.

Masked with chars
------------------------------
DATE [**********], [***********],  The driver's license [*********].

Masked with fixed length chars
------------------------------
DATE ****, ****,  The driver's license ****.

Obfuscated
------------------------------
DATE 07-04-1981, Dr Vivian Irving,  The driver's license K272344712994.
```

#### Add `.cache_folder` Parameter in `UpdateModels.updateCacheModels()`

This parameter lets user to define custom local paths for the folder on which pretrained models are saved (rather than using default cached_pretrained folder).

This cache_folder must be a path ("hdfs:..","file:...").

```bash
UpdateModels.updateCacheModels("file:/home/jsl/cache_pretrained_2")
```

```bash
UpdateModels.updateModels("12/01/2021","file:/home/jsl/cache_pretrained_2")
```

The cache folder used by default is the folder loaded in the spark configuration ` spark.jsl.settings.pretrained.cache_folder`.The default value for that property is `~/cache_pretrained`


#### S3 Access Credentials No Longer Shipped Along Licenses

S3 access credentials are no longer being shipped with licenses. Going forward, we'll use temporal S3 access credentials which will be periodically refreshed. All this will happen automatically and will be transparent to the user.
Still, for those users who would need to perform manual tasks involving access to S3, there's a mechanism to get access to the set of credentials being used by the library at any given time.

```bash
from sparknlp_jsl import get_credentials
get_credentials(spark)
```

#### Enhanced Security for the Library and log4shell Update

On top of periodical security checks on the library code, 3rd party dependencies were analyzed, and some dependencies reported as containing vulnerabilities were replaced by more secure options.
Also, the library was analyzed in the context of the recently discovered threat(CVE-2021-45105) on the log4j library. Spark NLP for Healthcare does not depend on the log4j library by itself, but the library gets loaded through some of its dependencies.
It's worth noting that the version of log4j dependency that will be in the classpath when running Spark NLP for Healthcare is 1.x, which would make the system vulnerable to CVE-2021-4104, instead of CVE-2021-45105. CVE-2021-4104 is related to the JMSAppender.
Spark NLP for Healthcare does not provide any log4j configuration, so it's up to the user to follow the recommendation of avoiding the use of the JMSAppender.


#### New Peer-Reviewed Conference Paper on Clinical Relation Extraction

We publish a new peer-reviewed conference paper titled [Deeper Clinical Document Understanding Using Relation Extraction](https://arxiv.org/pdf/2112.13259.pdf) explaining the applications of Relation Extraction in a text mining framework comprising of Named Entity Recognition (NER) and Relation Extraction (RE) models. The paper is accepted to SDU (Scientific Document Understanding) workshop at AAAI-2022 conference and claims new SOTA scores on 5 out of 7 Biomedical & Clinical Relation Extraction (RE) tasks.

|Dataset|FCNN|BioBERT|Curr-SOTA|
|-|-|-|-|
|i2b2-Temporal|68.7|**73.6**|72.41|
|i2b2-Clinical|60.4|**69.1**|67.97|
|DDI|69.2|72.1|**84.1**|
|CPI|65.8|74.3|**88.9**|
|PGR|81.2|**87.9**|79.4|
|ADE Corpus|89.2|**90.0**|83.7|
|Posology|87.8|**96.7**|96.1|

*Macro-averaged F1 scores of both RE models on public datasets. FCNN refers to the Speed-Optimized FCNN architecture, while BioBERT refers to the AccuracyOptimized BioBERT architecture. The SOTA metrics are obtained from (Guan et al. 2020), (Ningthoujam et al. 2019), (Asada, Miwa, and Sasaki 2020), (Phan et al. 2021), (Sousa
and Couto 2020), (Crone 2020), and (Yang et al. 2021) respectively.*


#### New Peer-Reviewed Conference Paper on Adverse Drug Events Extraction

We publish a new peer-reviewed conference paper titled [Mining Adverse Drug Reactions from Unstructured Mediums at Scale](https://arxiv.org/pdf/2201.01405.pdf) proposing an end-to-end Adverse Drug Event mining solution using Classification, NER, and Relation Extraction Models. The paper is accepted to W3PHIAI (INTERNATIONAL WORKSHOP ON HEALTH INTELLIGENCE) workshop at AAAI-2022 conference, and claims new SOTA scores on 1 benchmark dataset for Classification, 3 benchmark datasets for NER, and 1 benchmark dataset for Relation Extraction.

|Task | Dataset | Spark NLP | Curr-SOTA |
|-|-|-|-|
|Classification|ADE|85.96|**87.0**|
|Classification|CADEC|**86.69**|81.5|
|Entity Recognition|ADE|**91.75**|91.3|
|Entity Recognition|CADEC|**78.36**|71.9|
|Entity Recognition|SMM4H|**76.73**|67.81|
|Relation Extraction|ADE|**90.0**|83.7|

*All F1 scores are Macro-averaged*

#### New and Updated Notebooks

+ We have two new Notebooks:
  - [Chunk Sentence Splitter Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/18.Chunk_Sentence_Splitter.ipynb) that involves usage of `ChunkSentenceSplitter` annotator.
  - [Clinical Relation Extraction Spark NLP Paper Reproduce Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/10.3.Clinical_RE_SparkNLP_Paper_Reproduce.ipynb) that can be used for reproducing the results in  [Deeper Clinical Document Understanding Using Relation Extraction](https://arxiv.org/pdf/2112.13259.pdf) paper.

+ We have updated our existing notebooks by adding new features and functionalities. Here are updated notebooks:
  - [Clinical Named Entity Recognition Model](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb)
  - [Clinical Entity Resolver Models](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb)
  - [Clinical DeIdentification](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/4.Clinical_DeIdentification.ipynb)
  - [Clinical NER Chunk Merger](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/7.Clinical_NER_Chunk_Merger.ipynb)
  - [Pretrained Clinical Pipelines](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/11.Pretrained_Clinical_Pipelines.ipynb)
  - [Healthcare Code Mapping](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/11.1.Healthcare_Code_Mapping.ipynb)
  - [Improved Entity Resolvers in Spark NLP with sBert](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/24.Improved_Entity_Resolvers_in_SparkNLP_with_sBert.ipynb)


**To see more, please check : [Spark NLP Healthcare Workshop Repo](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings/Healthcare)**

<div class="prev_ver h3-box" markdown="1">

## Versions

</div>

<ul class="pagination">
    <li>
        <a href="release_notes_3_3_4">Version 3.3.4</a>
    </li>
    <li>
        <strong>Version 3.4.0</strong>
    </li>
    <li>
        <a href="release_notes_3_4_1">Version 3.4.1</a>
    </li>
</ul>

<ul class="pagination owl-carousel pagination_big">
    <li><a href="release_notes_4_2_1">4.2.1</a></li>
    <li><a href="release_notes_4_2_0">4.2.0</a></li>
    <li><a href="release_notes_4_1_0">4.1.0</a></li>
    <li><a href="release_notes_4_0_2">4.0.2</a></li>
    <li><a href="release_notes_4_0_0">4.0.0</a></li>
    <li><a href="release_notes_3_5_3">3.5.3</a></li>
    <li><a href="release_notes_3_5_2">3.5.2</a></li>
    <li><a href="release_notes_3_5_1">3.5.1</a></li>
    <li><a href="release_notes_3_5_0">3.5.0</a></li>
    <li><a href="release_notes_3_4_2">3.4.2</a></li>
    <li><a href="release_notes_3_4_1">3.4.1</a></li>
    <li class="active"><a href="release_notes_3_4_0">3.4.0</a></li>
    <li><a href="release_notes_3_3_4">3.3.4</a></li>
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