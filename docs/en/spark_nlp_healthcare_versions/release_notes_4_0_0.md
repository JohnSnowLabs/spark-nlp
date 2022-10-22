---
layout: docs
header: true
seotitle: Spark NLP for Healthcare | John Snow Labs
title: Spark NLP for Healthcare Release Notes 4.0.0
permalink: /docs/en/spark_nlp_healthcare_versions/release_notes_4_0_0
key: docs-licensed-release-notes
modify_date: 2021-07-14
show_nav: true
sidebar:
    nav: sparknlp-healthcare
---

## 4.0.0

#### Highlights

+ 8 new chunk mapper models and 9 new pretrained chunk mapper pipelines to convert one medical terminology to another (Snomed to ICD10, RxNorm to UMLS etc.)
+ 2 new medical NER models (`ner_clinical_trials_abstracts` and `ner_pathogen`) and pretrained NER pipelines
+ 20 new biomedical NER models based on the [LivingNER corpus](https://temu.bsc.es/livingner/2022/05/03/multilingual-corpus/) in **8 languages** (English, Spanish, French, Italian, Portuguese, Romanian, Catalan and Galician)
+ 2 new medical NER models for Romanian language (`ner_clinical`, `ner_clinical_bert`)
+ Deidentification support for **Romanian** language (`ner_deid_subentity`, `ner_deid_subentity_bert` and a pretrained deidentification pipeline)
+ The first public health model: Emotional stress classifier (`bert_sequence_classifier_stress`)
+ `ResolverMerger` annotator to merge the results of `ChunkMapperModel` and `SentenceEntityResolverModel` annotators
+ New Shortest Context Match and Token Index Features in `ContextualParserApproach`
+ Prettified relational categories in `ZeroShotRelationExtractionModel` annotator
+ Create graphs for open source `NerDLApproach` with the `TFGraphBuilder`
+ Spark NLP for Healthcare library installation with Poetry (dependency management and packaging tool)
+ Bug fixes
+ Updated notebooks
+ List of recently updated or added models (**50+ new medical models and pipelines**)



#### 8 New Chunk Mapper Models and 9 New Pretrained Chunk Mapper Pipelines to Convert One Medical Terminology to Another (Snomed to ICD10, RxNorm to UMLS etc.)

We are releasing **8 new `ChunkMapperModel` models and 9 new pretrained pipelines** for mapping clinical codes with their corresponding.

+ Mapper Models:

| Mapper Name           	| Source    	| Target    	|
|-----------------------	|-----------	|-----------	|
| [snomed_icd10cm_mapper](https://nlp.johnsnowlabs.com/2022/06/26/icd10cm_snomed_mapper_en_3_0.html) 	| SNOMED CT 	| ICD-10-CM 	|
| [icd10cm_snomed_mapper](https://nlp.johnsnowlabs.com/2022/06/26/icd10cm_snomed_mapper_en_3_0.html) 	| ICD-10-CM 	| SNOMED CT 	|
| [snomed_icdo_mapper](https://nlp.johnsnowlabs.com/2022/06/26/snomed_icdo_mapper_en_3_0.html)    	| SNOMED CT 	| ICD-O     	|
| [icdo_snomed_mapper](https://nlp.johnsnowlabs.com/2022/06/26/icdo_snomed_mapper_en_3_0.html)    	| ICD-O     	| SNOMED CT 	|
| [rxnorm_umls_mapper](https://nlp.johnsnowlabs.com/2022/06/26/rxnorm_umls_mapper_en_3_0.html)    	| RxNorm    	| UMLS      	|
| [icd10cm_umls_mapper](https://nlp.johnsnowlabs.com/2022/06/26/icd10cm_umls_mapper_en_3_0.html)   	| ICD-10-CM 	| UMLS      	|
| [mesh_umls_mapper](https://nlp.johnsnowlabs.com/2022/06/26/mesh_umls_mapper_en_3_0.html)      	| MeSH      	| UMLS      	|
| [snomed_umls_mapper](https://nlp.johnsnowlabs.com/2022/06/27/snomed_umls_mapper_en_3_0.html)    	| SNOMED CT 	| UMLS      	|

*Example*:

```python
...
snomed_resolver = SentenceEntityResolverModel.pretrained("sbertresolve_snomed_conditions", "en", "clinical/models") \
    .setInputCols(["ner_chunk", "sbert_embeddings"]) \
    .setOutputCol("snomed_code")\
    .setDistanceFunction("EUCLIDEAN")

chunkerMapper = ChunkMapperModel.pretrained("snomed_icd10cm_mapper", "en", "clinical/models")\
    .setInputCols(["snomed_code"])\
    .setOutputCol("icd10cm_mappings")\
    .setRels(["icd10cm_code"])

pipeline = PipelineModel(
    stages = [
        documentAssembler,
        sbert_embedder,
        snomed_resolver,
        chunkerMapper
        ])

light_pipeline= LightPipeline(pipeline)

result = light_pipeline.fullAnnotate("Radiating chest pain")

```

*Results* :

```bash
|    | ner_chunk            |   snomed_code | icd10cm_mappings   |
|---:|:---------------------|--------------:|:-------------------|
|  0 | Radiating chest pain |      10000006 | R07.9              |
```


+ Pretrained Pipelines:

| Pipeline Name          	| Source    	| Target    	|
|------------------------	|-----------	|-----------	|
| [icd10cm_snomed_mapping](https://nlp.johnsnowlabs.com/2022/06/27/icd10cm_snomed_mapping_en_3_0.html) 	| ICD-10-CM 	| SNOMED CT 	|
| [snomed_icd10cm_mapping](https://nlp.johnsnowlabs.com/2022/06/27/snomed_icd10cm_mapping_en_3_0.html) 	| SNOMED CT 	| ICD-10-CM 	|
| [icdo_snomed_mapping](https://nlp.johnsnowlabs.com/2022/06/27/icdo_snomed_mapping_en_3_0.html)    	| ICD-O     	| SNOMED CT 	|
| [snomed_icdo_mapping](https://nlp.johnsnowlabs.com/2022/06/27/snomed_icdo_mapping_en_3_0.html)    	| SNOMED CT 	| ICD-O     	|
| [rxnorm_ndc_mapping](https://nlp.johnsnowlabs.com/2022/06/27/rxnorm_ndc_mapping_en_3_0.html)     	| RxNorm    	| NDC       	|
| [icd10cm_umls_mapping](https://nlp.johnsnowlabs.com/2022/06/27/icd10cm_umls_mapping_en_3_0.html)   	| ICD-10-CM 	| UMLS      	|
| [mesh_umls_mapping](https://nlp.johnsnowlabs.com/2022/06/27/mesh_umls_mapping_en_3_0.html)      	| MeSH      	| UMLS      	|
| [rxnorm_umls_mapping](https://nlp.johnsnowlabs.com/2022/06/27/rxnorm_umls_mapping_en_3_0.html)    	| RxNorm    	| UMLS      	|
| [snomed_umls_mapping](https://nlp.johnsnowlabs.com/2022/06/27/snomed_umls_mapping_en_3_0.html)    	| SOMED CT  	| UMLS      	|


*Example*:

```python
from sparknlp.pretrained import PretrainedPipeline

pipeline= PretrainedPipeline("rxnorm_umls_mapping", "en", "clinical/models")
result= pipeline.annotate("1161611 315677")

```

*Results* :

```bash
{'document': ['1161611 315677'],
 'rxnorm_code': ['1161611', '315677'],
 'umls_code': ['C3215948', 'C0984912']}
```


#### 2 New Medical NER Models (`ner_clinical_trials_abstracts` and `ner_pathogene`) and Pretrained NER Pipelines

+ `ner_clinical_trials_abstracts`: This model can extract concepts related to clinical trial design, diseases, drugs, population, statistics and publication. It can detect `Age`, `AllocationRatio`, `Author`, `BioAndMedicalUnit`, `CTAnalysisApproach`, `CTDesign`, `Confidence`, `Country`, `DisorderOrSyndrome`, `DoseValue`, `Drug`, `DrugTime`, `Duration`, `Journal`, `NumberPatients`, `PMID`, `PValue`, `PercentagePatients`, `PublicationYear`, `TimePoint`, `Value` entities.

See [Model Hub Page](https://nlp.johnsnowlabs.com/2022/06/22/ner_clinical_trials_abstracts_en_3_0.html) for details.

*Example* :

```bash
...
clinical_ner = MedicalNerModel.pretrained("ner_clinical_trials_abstracts", "en", "clinical/models") \
      .setInputCols(["sentence", "token", "embeddings"]) \
      .setOutputCol("ner_tags")
...

sample_text = "A one-year, randomised, multicentre trial comparing insulin glargine with NPH insulin in combination with oral agents in patients with type 2 diabetes."
```

+ `bert_token_classifier_ner_clinical_trials_abstracts`: This model is the BERT-based version of `ner_clinical_trials_abstracts` model and it can detect `Age`, `AllocationRatio`, `Author`, `BioAndMedicalUnit`, `CTAnalysisApproach`, `CTDesign`, `Confidence`, `Country`, `DisorderOrSyndrome`, `DoseValue`, `Drug`, `DrugTime`, `Duration`, `Journal`, `NumberPatients`, `PMID`, `PValue`, `PercentagePatients`, `PublicationYear`, `TimePoint`, `Value` entities.

See [Model Hub Page](https://nlp.johnsnowlabs.com/2022/06/29/bert_token_classifier_ner_clinical_trials_abstracts_en_3_0.html) for details.

*Example* :

```python
...
tokenClassifier = MedicalBertForTokenClassifier.pretrained("bert_token_classifier_ner_clinical_trials_abstracts", "en", "clinical/models")\
       .setInputCols("token", "sentence")\
       .setOutputCol("ner")\
       .setCaseSensitive(True)
...

sample_text = "A one-year, randomised, multicentre trial comparing insulin glargine with NPH insulin in combination with oral agents in patients with type 2 diabetes."
```

+ `ner_clinical_trials_abstracts_pipeline`: This pretrained pipeline is build upon the `ner_clinical_trials_abstracts` model and it can extract `Age`, `AllocationRatio`, `Author`, `BioAndMedicalUnit`, `CTAnalysisApproach`, `CTDesign`, `Confidence`, `Country`, `DisorderOrSyndrome`, `DoseValue`, `Drug`, `DrugTime`, `Duration`, `Journal`, `NumberPatients`, `PMID`, `PValue`, `PercentagePatients`, `PublicationYear`, `TimePoint`, `Value` entities.

See[ Model Hub Page](https://nlp.johnsnowlabs.com/2022/06/27/ner_clinical_trials_abstracts_pipeline_en_3_0.html) for details.

*Example* :

```bash
pipeline = PretrainedPipeline("ner_clinical_trials_abstracts_pipeline", "en", "clinical/models")

result = pipeline.fullAnnotate("A one-year, randomised, multicentre trial comparing insulin glargine with NPH insulin in combination with oral agents in patients with type 2 diabetes.")
```

*Results* :

```bash
+----------------+------------------+
|           chunk|         ner_label|
+----------------+------------------+
|      randomised|          CTDesign|
|     multicentre|          CTDesign|
|insulin glargine|              Drug|
|     NPH insulin|              Drug|
| type 2 diabetes|DisorderOrSyndrome|
+----------------+------------------+
```

+ `ner_pathogen`: This model is trained for detecting medical conditions (influenza, headache, malaria, etc), medicine (aspirin, penicillin, methotrexate) and pathogenes (Corona Virus, Zika Virus, E. Coli, etc) in clinical texts. It can extract `Pathogen`, `MedicalCondition`, `Medicine` entities.

See [Model Hub Page](https://nlp.johnsnowlabs.com/2022/06/28/ner_pathogen_en_3_0.html) for details.

*Example* :

```bash
...
clinical_ner = MedicalNerModel.pretrained("ner_pathogen", "en", "clinical/models") \
      .setInputCols(["sentence", "token", "embeddings"]) \
      .setOutputCol("ner")
...

sample_text = "Racecadotril is an antisecretory medication and it has better tolerability than loperamide. Diarrhea is the condition of having loose, liquid or watery bowel movements each day. Signs of dehydration often begin with loss of the normal stretchiness of the skin. While it has been speculated that rabies virus, Lyssavirus and Ephemerovirus could be transmitted through aerosols, studies have concluded that this is only feasible in limited conditions."
```

+ `ner_pathogen_pipeline`: This pretrained pipeline is build upon the `ner_pathogen` model and it can extract  `Pathogen`, `MedicalCondition`, `Medicine` entities.

See [Model Hub Page](https://nlp.johnsnowlabs.com/2022/06/29/ner_pathogen_pipeline_en_3_0.html) for details.

*Example* :

```bash
pipeline = PretrainedPipeline("ner_pathogen_pipeline", "en", "clinical/models")

result = pipeline.fullAnnotate("Racecadotril is an antisecretory medication and it has better tolerability than loperamide. Diarrhea is the condition of having loose, liquid or watery bowel movements each day. Signs of dehydration often begin with loss of the normal stretchiness of the skin. While it has been speculated that rabies virus, Lyssavirus and Ephemerovirus could be transmitted through aerosols, studies have concluded that this is only feasible in limited conditions.")
```

*Results* :

```bash
+---------------+----------------+
|chunk          |ner_label       |
+---------------+----------------+
|Racecadotril   |Medicine        |
|loperamide     |Medicine        |
|Diarrhea       |MedicalCondition|
|dehydration    |MedicalCondition|
|rabies virus   |Pathogen        |
|Lyssavirus     |Pathogen        |
|Ephemerovirus  |Pathogen        |
+---------------+----------------+
```

+ `ner_biomedical_bc2gm_pipeline` : This pretrained pipeline can extract genes/proteins from medical texts by labelling them as `GENE_PROTEIN`.

See [Model Hub Page](https://nlp.johnsnowlabs.com/2022/06/22/ner_biomedical_bc2gm_pipeline_en_3_0.html) for details.

*Example* :

```python
pipeline = PretrainedPipeline("ner_biomedical_bc2gm_pipeline", "en", "clinical/models")

result = pipeline.fullAnnotate("""Immunohistochemical staining was positive for S-100 in all 9 cases stained, positive for HMB-45 in 9 (90%) of 10, and negative for cytokeratin in all 9 cases in which myxoid melanoma remained in the block after previous sections.""")
```

*Results* :

```bash
+-----------+------------+
|chunk      |ner_label   |
+-----------+------------+
|S-100      |GENE_PROTEIN|
|HMB-45     |GENE_PROTEIN|
|cytokeratin|GENE_PROTEIN|
+-----------+------------+
```

#### 20 New Biomedical NER Models Based on the [LivingNER corpus] in 8 Languages

+ We are releasing 20 new NER and `MedicalBertForTokenClassifier` models for **English, French, Italian, Portuguese, Romanian, Catalan and Galician* languages that are trained on the [LivingNER multilingual corpus](https://temu.bsc.es/livingner/2022/05/03/multilingual-corpus/) and for *Spanish* that is trained on [LivingNER corpus](https://temu.bsc.es/livingner/) is composed of clinical case reports extracted from miscellaneous medical specialties including COVID, oncology, infectious diseases, tropical medicine, urology, pediatrics, and others. These models can detect living species as `HUMAN` and `SPECIES` entities in clinical texts.

Here is the list of model names and their embeddings used while training:

| Language | Annotator                         | Embeddings                                  | Model Name                                        |
| -------- | --------------------------------- | ------------------------------------------- | ------------------------------------------------- |
| es       | MedicalBertForTokenClassification |                                             | [bert\_token\_classifier\_ner\_living\_species](https://nlp.johnsnowlabs.com/2022/06/27/bert_token_classifier_ner_living_species_es_3_0.html) |
| es       | MedicalNerModel                   | bert\_base\_cased\_es                       | [ner\_living\_species\_bert](https://nlp.johnsnowlabs.com/2022/06/22/ner_living_species_bert_es_3_0.html)                    |
| es       | MedicalNerModel                   | roberta\_base\_biomedical\_es               | [ner\_living\_species\_roberta](https://nlp.johnsnowlabs.com/2022/06/22/ner_living_species_roberta_es_3_0.html)                 |
| es       | MedicalNerModel                   | embeddings\_scielo\_300d\_es                | [ner\_living\_species\_300](https://nlp.johnsnowlabs.com/2022/06/22/ner_living_species_300_es_3_0.html)                     |
| es       | MedicalNerModel                   | w2v\_cc\_300d\_es                           | [ner\_living\_species](https://nlp.johnsnowlabs.com/2022/06/22/ner_living_species_es_3_0.html)                          |
| en       | MedicalBertForTokenClassification |                                             | [bert\_token\_classifier\_ner\_living\_species](https://nlp.johnsnowlabs.com/2022/06/26/bert_token_classifier_ner_living_species_en_3_0.html) |
| en       | MedicalNerModel                   | embeddings\_clinical\_en                    | [ner\_living\_species](https://nlp.johnsnowlabs.com/2022/06/22/ner_living_species_en_3_0.html)                          |
| en       | MedicalNerModel                   | biobert\_pubmed\_base\_cased\_en            | [ner\_living\_species\_biobert](https://nlp.johnsnowlabs.com/2022/06/22/ner_living_species_biobert_en_3_0.html)                 |
| fr       | MedicalNerModel                   | w2v\_cc\_300d\_fr                           | [ner\_living\_species](https://nlp.johnsnowlabs.com/2022/06/23/ner_living_species_fr_3_0.html)                          |
| fr       | MedicalNerModel                   | bert\_embeddings\_bert\_base\_fr\_cased     | [ner\_living\_species\_bert](https://nlp.johnsnowlabs.com/2022/06/23/ner_living_species_bert_fr_3_0.html)                    |
| pt       | MedicalBertForTokenClassification |                                             | [bert\_token\_classifier\_ner\_living\_species](https://nlp.johnsnowlabs.com/2022/06/27/bert_token_classifier_ner_living_species_pt_3_0.html) |
| pt       | MedicalNerModel                   | w2v\_cc\_300d\_pt                           | [ner\_living\_species](https://nlp.johnsnowlabs.com/2022/06/22/ner_living_species_pt_3_0.html)                          |
| pt       | MedicalNerModel                   | roberta\_embeddings\_BR\_BERTo\_pt          | [ner\_living\_species\_roberta](https://nlp.johnsnowlabs.com/2022/06/22/ner_living_species_roberta_pt_3_0.html)                 |
| pt       | MedicalNerModel                   | biobert\_embeddings\_biomedical\_pt         | [ner\_living\_species\_bert](https://nlp.johnsnowlabs.com/2022/06/22/ner_living_species_bert_pt_3_0.html)                    |
| it       | MedicalBertForTokenClassification |                                             | [bert\_token\_classifier\_ner\_living\_species](https://nlp.johnsnowlabs.com/2022/06/27/bert_token_classifier_ner_living_species_it_3_0.html) |
| it       | MedicalNerModel                   | bert\_base\_italian\_xxl\_cased\_it         | [ner\_living\_species\_bert](https://nlp.johnsnowlabs.com/2022/06/23/ner_living_species_bert_it_3_0.html)                    |
| it       | MedicalNerModel                   | w2v\_cc\_300d\_it                           | [ner\_living\_species](https://nlp.johnsnowlabs.com/2022/06/23/ner_living_species_it_3_0.html)                          |
| ro       | MedicalNerModel                   | bert\_base\_cased\_ro                       | [ner\_living\_species\_bert](https://nlp.johnsnowlabs.com/2022/06/23/ner_living_species_bert_ro_3_0.html)                    |
| cat      | MedicalNerModel                   | w2v\_cc\_300d\_cat                          | [ner\_living\_species](https://nlp.johnsnowlabs.com/2022/06/23/ner_living_species_ca_3_0.html)                          |
| gal      | MedicalNerModel                   | w2v\_cc\_300d\_gal                          | [ner\_living\_species](https://nlp.johnsnowlabs.com/2022/06/23/ner_living_species_gl_3_0.html)                          |

*Example* :

```bash
...
clinical_ner = MedicalNerModel.pretrained("ner_living_species", "en", "clinical/models") \
      .setInputCols(["sentence", "token", "embeddings"]) \
      .setOutputCol("ner_tags")
...

results = ner_model.transform(spark.createDataFrame([["""Patient aged 61 years; no known drug allergies, smoker of 63 packs/year, significant active alcoholism, recently diagnosed hypertension. He came to the emergency department approximately 4 days ago with a frontal headache coinciding with a diagnosis of hypertension, for which he was started on antihypertensive treatment. The family reported that they found him "slower" accompanied by behavioural alterations; with no other accompanying symptoms.Physical examination: Glasgow Glasgow 15; neurological examination without focality except for bradypsychia and disorientation in time, person and space. Afebrile. BP: 159/92; heart rate 70 and O2 Sat: 93%; abdominal examination revealed hepatomegaly of two finger widths with no other noteworthy findings. CBC: Legionella antigen and pneumococcus in urine negative."""]], ["text"]))
```

*Results* :

```bash
+------------+-------+
|ner_chunk   |label  |
+------------+-------+
|Patient     |HUMAN  |
|family      |HUMAN  |
|person      |HUMAN  |
|Legionella  |SPECIES|
|pneumococcus|SPECIES|
+------------+-------+
```

#### 2 New Medical NER Models for Romanian Language

We trained `ner_clinical` and `ner_clinical_bert` models that can detect `Measurements`, `Form`, `Symptom`, `Route`, `Procedure`, `Disease_Syndrome_Disorder`, `Score`, `Drug_Ingredient`, `Pulse`, `Frequency`, `Date`, `Body_Part`, `Drug_Brand_Name`, `Time`, `Direction`, `Dosage`, `Medical_Device`, `Imaging_Technique`, `Test`, `Imaging_Findings`, `Imaging_Test`, `Test_Result`, `Weight`, `Clinical_Dept` and `Units` entities in Romanian clinical texts.

+ `ner_clinical`: This model is trained with `w2v_cc_300d` embeddings model.

*Example* :

```python
...
embeddings = WordEmbeddingsModel.pretrained("w2v_cc_300d","ro")\
        .setInputCols(["sentence","token"])\
        .setOutputCol("word_embeddings")

clinical_ner = MedicalNerModel.pretrained("ner_clinical", "ro", "clinical/models")\
        .setInputCols(["sentence","token","word_embeddings"])\
        .setOutputCol("ner")
...

sample_text = "Aorta ascendenta inlocuita cu proteza de Dacron de la nivelul anulusului pana pe segmentul ascendent distal pe o lungime aproximativa de 75 mm."
```

+ `ner_clinical_bert`: This model is trained with `bert_base_cased` embeddings model.

*Example* :

 ```python
 ...
 embeddings = BertEmbeddings.pretrained("bert_base_cased", "ro")\
        .setInputCols(["sentence","token"])\
        .setOutputCol("word_embeddings")

clinical_ner = MedicalNerModel.pretrained("ner_clinical_bert", "ro", "clinical/models")\
        .setInputCols(["sentence","token","word_embeddings"])\
        .setOutputCol("ner")
...

sample_text = "Aorta ascendenta inlocuita cu proteza de Dacron de la nivelul anulusului pana pe segmentul ascendent distal pe o lungime aproximativa de 75 mm."
```

*Results* :

```bash
+-------------------+--------------+
|             chunks|      entities|
+-------------------+--------------+
|   Aorta ascendenta|     Body_Part|
|  proteza de Dacron|Medical_Device|
|         anulusului|     Body_Part|
|segmentul ascendent|     Body_Part|
|             distal|     Direction|
|                 75|  Measurements|
|                 mm|         Units|
+-------------------+--------------+
```


####  Deidentification Support for Romanian Language (`ner_deid_subentity`, `ner_deid_subentity_bert` and a Pretrained Deidentification Pipeline)

We trained two new NER models to find PHI data (protected health information) that may need to be deidentified in **Romanian**. `ner_deid_subentity` and `ner_deid_subentity_bert` models are trained with in-house annotations and can detect 17 different entities (`AGE`, `CITY`, `COUNTRY`, `DATE`, `DOCTOR`, `EMAIL`, `FAX`, `HOSPITAL`, `IDNUM`, `LOCATION-OTHER`, `MEDICALRECORD`, `ORGANIZATION`, `PATIENT`, `PHONE`, `PROFESSION`, `STREET`, `ZIP`).

+ `ner_deid_subentity`: This model is trained with `w2v_cc_300d` embeddings model.

See [Model Hub Page](https://nlp.johnsnowlabs.com/2022/06/27/ner_deid_w2v_subentity_ro_3_0.html) for details.

*Example* :

```python
...
embeddings = WordEmbeddingsModel.pretrained("w2v_cc_300d","ro")\
        .setInputCols(["sentence","token"])\
        .setOutputCol("word_embeddings")

clinical_ner = MedicalNerModel.pretrained("ner_deid_subentity", "ro", "clinical/models")\
        .setInputCols(["sentence","token","word_embeddings"])\
        .setOutputCol("ner")
...

sample_text = """
Spitalul Pentru Ochi de Deal, Drumul Oprea Nr. 972 Vaslui, 737405 România
Tel: +40(235)413773
Data setului de analize: 25 May 2022 15:36:00
Nume si Prenume : BUREAN MARIA, Varsta: 77
Medic : Agota Evelyn Tımar
C.N.P : 2450502264401"""
```

+ `ner_deid_subentity_bert`: This model is trained with `bert_base_cased` embeddings model.

See [Model Hub Page](https://nlp.johnsnowlabs.com/2022/06/27/ner_deid_bert_subentity_ro_3_0.html) for details.

*Example* :

 ```python
 ...
 embeddings = BertEmbeddings.pretrained("bert_base_cased", "ro")\
        .setInputCols(["sentence","token"])\
        .setOutputCol("word_embeddings")

clinical_ner = MedicalNerModel.pretrained("ner_deid_subentity_bert", "ro", "clinical/models")\
        .setInputCols(["sentence","token","word_embeddings"])\
        .setOutputCol("ner")
...

text = """
Spitalul Pentru Ochi de Deal, Drumul Oprea Nr. 972 Vaslui, 737405 România
Tel: +40(235)413773
Data setului de analize: 25 May 2022 15:36:00
Nume si Prenume : BUREAN MARIA, Varsta: 77
Medic : Agota Evelyn Tımar
C.N.P : 2450502264401"""
```

*Results* :

```bash
+----------------------------+---------+
|chunk                       |ner_label|
+----------------------------+---------+
|Spitalul Pentru Ochi de Deal|HOSPITAL |
|Drumul Oprea Nr             |STREET   |
|Vaslui                      |CITY     |
|737405                      |ZIP      |
|+40(235)413773              |PHONE    |
|25 May 2022                 |DATE     |
|BUREAN MARIA                |PATIENT  |
|77                          |AGE      |
|Agota Evelyn Tımar          |DOCTOR   |
|2450502264401               |IDNUM    |
+----------------------------+---------+
```

+ `clinical_deidentification`: This pretrained pipeline that can be used to deidentify PHI information from Romanian medical texts. The PHI information will be masked and obfuscated in the resulting text. The pipeline can mask and obfuscate `ACCOUNT`, `PLATE`, `LICENSE`, `AGE`, `CITY`, `COUNTRY`, `DATE`, `DOCTOR`, `EMAIL`, `FAX`, `HOSPITAL`, `IDNUM`, `LOCATION-OTHER`, `MEDICALRECORD`, `ORGANIZATION`, `PATIENT`, `PHONE`, `PROFESSION`, `STREET`, `ZIP` entities.

See [Model Hub Page](https://nlp.johnsnowlabs.com/2022/06/28/clinical_deidentification_ro_3_0.html) for details.

*Example* :

```python
from sparknlp.pretrained import PretrainedPipeline
deid_pipeline = PretrainedPipeline("clinical_deidentification", "ro", "clinical/models")

text = "Varsta : 77, Nume si Prenume : BUREAN MARIA, Data setului de analize: 25 May 2022, Licență : B004256985M, Înmatriculare : CD205113, Cont : FXHZ7170951927104999"

result = deid_pipeline.annotate(text)

print("\nMasked with entity labels")
print("-"*30)
print("\n".join(result['masked']))
print("\nMasked with chars")
print("-"*30)
print("\n".join(result['masked_with_chars']))
print("\nMasked with fixed length chars")
print("-"*30)
print("\n".join(result['masked_fixed_length_chars']))
print("\nObfuscated")
print("-"*30)
print("\n".join(result['obfuscated']))
```

*Results* :

```bash
Masked with entity labels
------------------------------
Varsta : <AGE>, Nume si Prenume : <PATIENT>, Data setului de analize: <DATE>, Licență : <LICENSE>, Înmatriculare : <PLATE>, Cont : <ACCOUNT>

Masked with chars
------------------------------
Varsta : **, Nume si Prenume : [**********], Data setului de analize: [*********], Licență : [*********], Înmatriculare : [******], Cont : [******************]

Masked with fixed length chars
------------------------------
Varsta : ****, Nume si Prenume : ****, Data setului de analize: ****, Licență : ****, Înmatriculare : ****, Cont : ****

Obfuscated
------------------------------
Varsta : 91, Nume si Prenume : Dragomir Emilia, Data setului de analize: 01-04-2001, Licență : T003485962M, Înmatriculare : AR-65-UPQ, Cont : KHHO5029180812813651
```

#### The First Public Health Model: Emotional Stress Classifier

We are releasing a new  `bert_sequence_classifier_stress` model that can classify whether the content of a text expresses emotional stress. It is a [PHS-BERT-based](https://huggingface.co/publichealthsurveillance/PHS-BERT) model and trained with the [Dreaddit dataset](https://arxiv.org/abs/1911.00133).

*Example* :

```python
...
sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_stress", "en", "clinical/models")\
    .setInputCols(["document","token"])\
    .setOutputCol("class")

sample_text = "No place in my city has shelter space for us, and I won't put my baby on the literal street. What cities have good shelter programs for homeless mothers and children?"
```

*Results* :

```bash
+----------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------+
|text                                                                                                                                                                  |   class|
+----------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------+
|No place in my city has shelter space for us, and I won't put my baby on the literal street. What cities have good shelter programs for homeless mothers and children?|[stress]|
+----------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------+
```

#### `ResolverMerger` Annotator to Merge the Results of `ChunkMapperModel` and `SentenceEntityResolverModel` Annotators

`ResolverMerger` annotator allows to merge the results of `ChunkMapperModel` and `SentenceEntityResolverModel` annotators. You can detect your results that fail by `ChunkMapperModel` with `ChunkMapperFilterer` and then merge your resolver and mapper results with `ResolverMerger`.

*Example* :

```python
...
chunkerMapper = ChunkMapperModel.pretrained("rxnorm_mapper", "en", "clinical/models")\
      .setInputCols(["chunk"])\
      .setOutputCol("RxNorm_Mapper")\
      .setRel("rxnorm_code")

cfModel = ChunkMapperFilterer() \
    .setInputCols(["chunk", "RxNorm_Mapper"]) \
    .setOutputCol("chunks_fail") \
    .setReturnCriteria("fail")
...
resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_rxnorm_augmented", "en", "clinical/models") \
    .setInputCols(["chunks_fail", "sentence_embeddings"]) \
    .setOutputCol("resolver_code") \
    .setDistanceFunction("EUCLIDEAN")

resolverMerger = ResolverMerger()\
    .setInputCols(["resolver_code","RxNorm_Mapper"])\
    .setOutputCol("RxNorm")
...
```

*Results* :

```bash
+--------------------------------+-----------------------+---------------+-------------+-------------------------+
|chunk                           |RxNorm_Mapper          |chunks_fail    |resolver_code|RxNorm                   |
+--------------------------------+-----------------------+---------------+-------------+-------------------------+
|[Adapin 10 MG, coumadin 5 mg]   |[1000049, NONE]        |[coumadin 5 mg]|[855333]     |[1000049, 855333]        |
|[Avandia 4 mg, Tegretol, zytiga]|[NONE, 203029, 1100076]|[Avandia 4 mg] |[261242]     |[261242, 203029, 1100076]|
+--------------------------------+-----------------------+---------------+-------------+-------------------------+
```

#### New Shortest Context Match and Token Index Features in `ContextualParserApproach`

We have new functionalities in `ContextualParserApproach` to make it more performant.

+ `setShortestContextMatch()` parameter will allow stop looking for matches in the text when a token defined as a suffix is found. Also it will keep tracking of the last mathced `prefix` and subsequent mathches with `suffix`.

+ Now the index of the matched token can be found in metadata.


*Example* :
```python
...
contextual_parser = ContextualParserApproach() \
    .setInputCols(["sentence", "token"])\
    .setOutputCol("entity")\
    .setJsonPath("cities.json")\
    .setCaseSensitive(True)\
    .setDictionary('cities.tsv', options={"orientation":"vertical"})\
    .setShortestContextMatch(True)
...

sample_text = "Peter Parker is a nice guy and lives in Chicago."
```

*Results* :

```bash
+-------+---------+----------+
|chunk  |ner_label|tokenIndex|
+-------+---------+----------+
|Chicago|City     |9         |
+-------+---------+----------+
```


#### Prettified relational categories in `ZeroShotRelationExtractionModel` annotator

Now you can `setRelationalCategories()` between the entity labels by using a single `{}` instead of two.

*Example* :

```python
re_model = ZeroShotRelationExtractionModel.pretrained("re_zeroshot_biobert", "en", "clinical/models")\
    .setInputCols(["re_ner_chunks", "sentences"]) \
    .setOutputCol("relations")\
    .setRelationalCategories({"ADE": ["{DRUG} causes {PROBLEM}."]})
```

#### Create Graphs for Open Source `NerDLApproach` with the `TFGraphBuilder`

Now you can create graphs for model training with `NerDLApproach` by using the new `setIsMedical()` parameter of `TFGraphBuilder` annotator. If `setIsMedical(True)`, the model can be trained with `MedicalNerApproach`, but if it is `setIsMedical(False)` it can be used with `NerDLApproach` for training non-medical models.

```python
graph_folder_path = "./graphs"

ner_graph_builder = TFGraphBuilder()\
    .setModelName("ner_dl")\
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setLabelColumn("label")\
    .setGraphFile("auto")\
    .setHiddenUnitsNumber(20)\
    .setGraphFolder(graph_folder_path)\
    .setIsMedical(False)

ner = NerDLApproach() \
    ...
    .setGraphFolder(graph_folder_path)

ner_pipeline = Pipeline()([
    ...,
    ner_graph_builder,
    ner
    ])
```


#### Spark NLP for Healthcare Library Installation with Poetry Documentation (dependency management and packaging tool).

We have a new documentation page for showing Spark NLP for Healthcare installation with Poetry. You can find it [here](https://nlp.johnsnowlabs.com/docs/en/licensed_install#install-with-poetry).


#### Bug fixes
+ `ContextualParserApproach`: Fixed the bug using a dictionary and document rule scope in JSON config file.
+ `RENerChunksFilter`: Preparing a pretrained pipeline with `RENerChunksFilter` annotator issue is fixed.


#### Updated Notebooks

+ [ZeroShot Clinical Relation Extraction Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/10.3.ZeroShot_Clinical_Relation_Extraction.ipynb):  Added new features, visualization and new examples.
+ [Clinical_Entity_Resolvers Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb): Added an example of `ResolverMerger`.
+ [Chunk Mapping Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/26.Chunk_Mapping.ipynb): Added new models into the model list and an example of mapper pretrained pipelines.
+ [Healthcare Code Mapping Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/11.1.Healthcare_Code_Mapping.ipynb): Added all mapper pretrained pipeline examples.



#### List of Recently Updated and Added Models

- `ner_pathogene`
- `ner_pathogen_pipeline`
- `ner_clinical_trials_abstracts`
- `bert_token_classifier_ner_clinical_trials_abstracts`
- `ner_clinical_trials_abstracts_pipeline`
- `ner_biomedical_bc2gm_pipeline`
- `bert_sequence_classifier_stress`
- `icd10cm_snomed_mapper`
- `snomed_icd10cm_mapper`
- `snomed_icdo_mapper`
- `icdo_snomed_mapper`
- `rxnorm_umls_mapper`
- `icd10cm_umls_mapper`
- `mesh_umls_mapper`
- `snomed_umls_mapper`
- `icd10cm_snomed_mapping`
- `snomed_icd10cm_mapping`
- `icdo_snomed_mapping`
- `snomed_icdo_mapping`
- `rxnorm_ndc_mapping`
- `icd10cm_umls_mapping`
- `mesh_umls_mapping`
- `rxnorm_umls_mapping`
- `snomed_umls_mapping`
- `drug_action_tretment_mapper`
- `normalized_section_header_mapper`
- `drug_brandname_ndc_mapper`
- `abbreviation_mapper`
- `rxnorm_ndc_mapper`
- `rxnorm_action_treatment_mapper`
- `rxnorm_mapper`
- `ner_deid_subentity` -> `ro`
- `ner_deid_subentity_bert` -> `ro`
- `clinical_deidentification` -> `ro`
- `ner_clinical` -> `ro`
- `ner_clinical_bert` -> `ro`
- `bert_token_classifier_ner_living_species` -> `es`
- `ner_living_species_bert` -> `es`
- `ner_living_species_roberta` -> `es`
- `ner_living_species_300` -> `es`
- `ner_living_species` -> `es`
- `bert_token_classifier_ner_living_species` -> `en`
- `ner_living_species` -> `en`
- `ner_living_species_biobert` -> `en`
- `ner_living_species` -> `fr`
- `ner_living_species_bert` -> `fr`
- `bert_token_classifier_ner_living_species` -> `pt`
- `ner_living_species` -> `pt`
- `ner_living_species_roberta` -> `pt`
- `ner_living_species_bert` -> `pt`
- `bert_token_classifier_ner_living_species` -> `it`
- `ner_living_species_bert` -> `it`
- `ner_living_species` -> `pt`
- `ner_living_species_bert` -> `ro`
- `ner_living_species` -> `ro`
- `ner_living_species` -> `gal`

For all Spark NLP for healthcare models, please check: [Models Hub Page](https://nlp.johnsnowlabs.com/models?edition=Spark+NLP+for+Healthcare)

<div class="prev_ver h3-box" markdown="1">

## Versions

</div>

<ul class="pagination">
    <li>
        <a href="release_notes_3_5_3">Version 3.5.3</a>
    </li>
    <li>
        <strong>Version 4.0.0</strong>
    </li>
    <li>
        <a href="release_notes_4_0_2">Version 4.0.2</a>
    </li>
</ul>

<ul class="pagination owl-carousel pagination_big">
    <li><a href="release_notes_4_2_1">4.2.1</a></li>
    <li><a href="release_notes_4_2_0">4.2.0</a></li>
    <li><a href="release_notes_4_1_0">4.1.0</a></li>
    <li><a href="release_notes_4_0_2">4.0.2</a></li>
    <li class="active"><a href="release_notes_4_0_0">4.0.0</a></li>
    <li><a href="release_notes_3_5_3">3.5.3</a></li>
    <li><a href="release_notes_3_5_2">3.5.2</a></li>
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