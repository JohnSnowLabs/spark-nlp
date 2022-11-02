---
layout: docs
header: true
seotitle: Spark NLP for Healthcare | John Snow Labs
title: Spark NLP for Healthcare Release Notes 3.3.0
permalink: /docs/en/spark_nlp_healthcare_versions/release_notes_3_3_0
key: docs-licensed-release-notes
modify_date: 2021-07-14
show_nav: true
sidebar:
    nav: sparknlp-healthcare
---

## 3.3.0
We are glad to announce that Spark NLP Healthcare 3.3.0 has been released!.

#### Highlights
+ NER Finder Pretrained Pipelines to Run Run 48 different Clinical NER and 21 Different Biobert Models At Once Over the Input Text
+ 3 New Sentence Entity Resolver Models (3-char ICD10CM, RxNorm_NDC, HCPCS)
+ Updated UMLS Entity Resolvers (Dropping Invalid Codes)
+ 5 New Clinical NER Models (Trained By BertForTokenClassification Approach)
+ Radiology NER Model Trained On cheXpert Dataset
+ New Speed Benchmarks on Databricks
+ NerConverterInternal Fixes
+ Simplified Setup and Recommended Use of start() Function
+ NER Evaluation Metrics Fix
+ New Notebooks (Including How to Use SparkNLP with Neo4J)

#### NER Finder Pretrained Pipelines to Run Run 48 different Clinical NER and 21 Different Biobert Models At Once Over the Input Text

We are releasing two new NER Pretrained Pipelines that can be used to explore all the available pretrained NER models at once. You can check [NER Profiling Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/11.2.Pretrained_NER_Profiling_Pipelines.ipynb) to see how to use these pretrained pipelines.

- `ner_profiling_clinical` : When you run this pipeline over your text, you will end up with the predictions coming out of each of the 48 pretrained clinical NER models trained with `embeddings_clinical`.

|Clinical NER Model List|
|-|
|ner_ade_clinical|
|ner_posology_greedy|
|ner_risk_factors|
|jsl_ner_wip_clinical|
|ner_human_phenotype_gene_clinical|
|jsl_ner_wip_greedy_clinical|
|ner_cellular|
|ner_cancer_genetics|
|jsl_ner_wip_modifier_clinical|
|ner_drugs_greedy|
|ner_deid_sd_large|
|ner_diseases|
|nerdl_tumour_demo|
|ner_deid_subentity_augmented|
|ner_jsl_enriched|
|ner_genetic_variants|
|ner_bionlp|
|ner_measurements_clinical|
|ner_diseases_large|
|ner_radiology|
|ner_deid_augmented|
|ner_anatomy|
|ner_chemprot_clinical|
|ner_posology_experimental|
|ner_drugs|
|ner_deid_sd|
|ner_posology_large|
|ner_deid_large|
|ner_posology|
|ner_deidentify_dl|
|ner_deid_enriched|
|ner_bacterial_species|
|ner_drugs_large|
|ner_clinical_large|
|jsl_rd_ner_wip_greedy_clinical|
|ner_medmentions_coarse|
|ner_radiology_wip_clinical|
|ner_clinical|
|ner_chemicals|
|ner_deid_synthetic|
|ner_events_clinical|
|ner_posology_small|
|ner_anatomy_coarse|
|ner_human_phenotype_go_clinical|
|ner_jsl_slim|
|ner_jsl|
|ner_jsl_greedy|
|ner_events_admission_clinical|


- `ner_profiling_biobert` : When you run this pipeline over your text, you will end up with the predictions coming out of each of the 21 pretrained clinical NER models trained with `biobert_pubmed_base_cased`.

|BioBert NER Model List|
|-|
|ner_cellular_biobert|
|ner_diseases_biobert|
|ner_events_biobert|
|ner_bionlp_biobert|
|ner_jsl_greedy_biobert|
|ner_jsl_biobert|
|ner_anatomy_biobert|
|ner_jsl_enriched_biobert|
|ner_human_phenotype_go_biobert|
|ner_deid_biobert|
|ner_deid_enriched_biobert|
|ner_clinical_biobert|
|ner_anatomy_coarse_biobert|
|ner_human_phenotype_gene_biobert|
|ner_posology_large_biobert|
|jsl_rd_ner_wip_greedy_biobert|
|ner_posology_biobert|
|jsl_ner_wip_greedy_biobert|
|ner_chemprot_biobert|
|ner_ade_biobert|
|ner_risk_factors_biobert|

You can also check [Models Hub](https://nlp.johnsnowlabs.com/models) page for more information about all these NER models and more.

*Example* :

```
from sparknlp.pretrained import PretrainedPipeline
ner_profiling_pipeline = PretrainedPipeline('ner_profiling_biobert', 'en', 'clinical/models')

result = ner_profiling_pipeline.annotate("A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus ( T2DM ), one prior episode of HTG-induced pancreatitis three years prior to presentation , associated with an acute hepatitis , and obesity with a body mass index ( BMI ) of 33.5 kg/m2 , presented with a one-week history of polyuria , polydipsia , poor appetite , and vomiting .")
```

*Results* :

```bash
sentence :  ['A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus ( T2DM ), one prior episode of HTG-induced pancreatitis three years prior to presentation , associated with an acute hepatitis , and obesity with a body mass index ( BMI ) of 33.5 kg/m2 , presented with a one-week history of polyuria , polydipsia , poor appetite , and vomiting .']
token :  ['A', '28-year-old', 'female', 'with', 'a', 'history', 'of', 'gestational', 'diabetes', 'mellitus', 'diagnosed', 'eight', 'years', 'prior', 'to', 'presentation', 'and', 'subsequent', 'type', 'two', 'diabetes', 'mellitus', '(', 'T2DM', '),', 'one', 'prior', 'episode', 'of', 'HTG-induced', 'pancreatitis', 'three', 'years', 'prior', 'to', 'presentation', ',', 'associated', 'with', 'an', 'acute', 'hepatitis', ',', 'and', 'obesity', 'with', 'a', 'body', 'mass', 'index', '(', 'BMI', ')', 'of', '33.5', 'kg/m2', ',', 'presented', 'with', 'a', 'one-week', 'history', 'of', 'polyuria', ',', 'polydipsia', ',', 'poor', 'appetite', ',', 'and', 'vomiting', '.']
ner_cellular_biobert_chunks :  []
ner_diseases_biobert_chunks :  ['gestational diabetes mellitus', 'type two diabetes mellitus', 'T2DM', 'HTG-induced pancreatitis', 'hepatitis', 'obesity', 'polyuria', 'polydipsia', 'poor appetite', 'vomiting']
ner_events_biobert_chunks :  ['gestational diabetes mellitus', 'eight years', 'presentation', 'type two diabetes mellitus ( T2DM', 'HTG-induced pancreatitis', 'three years', 'presentation', 'an acute hepatitis', 'obesity', 'a body mass index', 'BMI', 'presented', 'a one-week', 'polyuria', 'polydipsia', 'poor appetite', 'vomiting']
ner_bionlp_biobert_chunks :  []
ner_jsl_greedy_biobert_chunks :  ['28-year-old', 'female', 'gestational diabetes mellitus', 'eight years prior', 'type two diabetes mellitus', 'T2DM', 'HTG-induced pancreatitis', 'three years prior', 'acute hepatitis', 'obesity', 'body mass index', 'BMI ) of 33.5 kg/m2', 'one-week', 'polyuria', 'polydipsia', 'poor appetite', 'vomiting']
ner_jsl_biobert_chunks :  ['28-year-old', 'female', 'gestational diabetes mellitus', 'eight years prior', 'type two diabetes mellitus', 'T2DM', 'HTG-induced pancreatitis', 'three years prior', 'acute', 'hepatitis', 'obesity', 'body mass index', 'BMI ) of 33.5 kg/m2', 'one-week', 'polyuria', 'polydipsia', 'poor appetite', 'vomiting']
ner_anatomy_biobert_chunks :  ['body']
ner_jsl_enriched_biobert_chunks :  ['28-year-old', 'female', 'gestational diabetes mellitus', 'type two diabetes mellitus', 'T2DM', 'HTG-induced pancreatitis', 'acute', 'hepatitis', 'obesity', 'polyuria', 'polydipsia', 'poor appetite', 'vomiting']
ner_human_phenotype_go_biobert_chunks :  ['obesity', 'polyuria', 'polydipsia']
ner_deid_biobert_chunks :  ['eight years', 'three years']
ner_deid_enriched_biobert_chunks :  []
ner_clinical_biobert_chunks :  ['gestational diabetes mellitus', 'subsequent type two diabetes mellitus ( T2DM', 'HTG-induced pancreatitis', 'an acute hepatitis', 'obesity', 'a body mass index ( BMI )', 'polyuria', 'polydipsia', 'poor appetite', 'vomiting']
ner_anatomy_coarse_biobert_chunks :  ['body']
ner_human_phenotype_gene_biobert_chunks :  ['obesity', 'mass', 'polyuria', 'polydipsia', 'vomiting']
ner_posology_large_biobert_chunks :  []
jsl_rd_ner_wip_greedy_biobert_chunks :  ['gestational diabetes mellitus', 'type two diabetes mellitus', 'T2DM', 'HTG-induced pancreatitis', 'acute hepatitis', 'obesity', 'body mass index', '33.5', 'kg/m2', 'polyuria', 'polydipsia', 'poor appetite', 'vomiting']
ner_posology_biobert_chunks :  []
jsl_ner_wip_greedy_biobert_chunks :  ['28-year-old', 'female', 'gestational diabetes mellitus', 'eight years prior', 'type two diabetes mellitus', 'T2DM', 'HTG-induced pancreatitis', 'three years prior', 'acute hepatitis', 'obesity', 'body mass index', 'BMI ) of 33.5 kg/m2', 'one-week', 'polyuria', 'polydipsia', 'poor appetite', 'vomiting']
ner_chemprot_biobert_chunks :  []
ner_ade_biobert_chunks :  ['pancreatitis', 'acute hepatitis', 'polyuria', 'polydipsia', 'poor appetite', 'vomiting']
ner_risk_factors_biobert_chunks :  ['diabetes mellitus', 'subsequent type two diabetes mellitus', 'obesity']
```

#### 3 New Sentence Entity Resolver Models (3-char ICD10CM, RxNorm_NDC, HCPCS)

+ `sbiobertresolve_hcpcs` : This model maps extracted medical entities to [Healthcare Common Procedure Coding System (HCPCS)](https://www.nlm.nih.gov/research/umls/sourcereleasedocs/current/HCPCS/index.html#:~:text=The%20Healthcare%20Common%20Procedure%20Coding,%2C%20supplies%2C%20products%20and%20services.)
 codes using `sbiobert_base_cased_mli` sentence embeddings. It also returns the domain information of the codes in the `all_k_aux_labels` parameter in the metadata of the result.

*Example* :

```bash
documentAssembler = DocumentAssembler()\
      .setInputCol("text")\
      .setOutputCol("ner_chunk")
sbert_embedder = BertSentenceEmbeddings.pretrained('sbiobert_base_cased_mli', 'en','clinical/models')\
      .setInputCols(["ner_chunk"])\
      .setOutputCol("sentence_embeddings")

hcpcs_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_hcpcs", "en", "clinical/models") \
      .setInputCols(["ner_chunk", "sentence_embeddings"]) \
      .setOutputCol("hcpcs_code")\
      .setDistanceFunction("EUCLIDEAN")
hcpcs_pipelineModel = PipelineModel(
    stages = [
        documentAssembler,
        sbert_embedder,
        hcpcs_resolver])

res = hcpcs_pipelineModel.transform(spark.createDataFrame([["Breast prosthesis, mastectomy bra, with integrated breast prosthesis form, unilateral, any size, any type"]]).toDF("text"))
```

*Results* :

|ner_chunk|hcpcs_code|all_codes|all_resolutions|domain|
|-|-|-|-|-|
| Breast prosthesis, mastectomy bra, with integrated breast prosthesis form, unilateral, any size, any type  | L8001  |[L8001, L8002, L8000, L8033, L8032, ...]   |'Breast prosthesis, mastectomy bra, with integrated breast prosthesis form, unilateral, any size, any type', 'Breast prosthesis, mastectomy bra, with integrated breast prosthesis form, bilateral, any size, any type', 'Breast prosthesis, mastectomy bra, without integrated breast prosthesis form, any size, any type', 'Nipple prosthesis, custom fabricated, reusable, any material, any type, each', ...  | Device, Device, Device, Device, Device, ...  |


+ `sbiobertresolve_icd10cm_generalised` : This model maps medical entities to 3 digit ICD10CM codes (according to ICD10 code structure the first three characters represent general type of the injury or disease). Difference in results (compared with `sbiobertresolve_icd10cm`) can be observed in the example below.

*Example* :

```bash
documentAssembler = DocumentAssembler()\
      .setInputCol("text")\
      .setOutputCol("ner_chunk")
sbert_embedder = BertSentenceEmbeddings.pretrained('sbiobert_base_cased_mli', 'en','clinical/models')\
      .setInputCols(["ner_chunk"])\
      .setOutputCol("sentence_embeddings")

icd_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_icd10cm_generalised", "en", "clinical/models") \
      .setInputCols(["ner_chunk", "sentence_embeddings"]) \
      .setOutputCol("icd_code")\
      .setDistanceFunction("EUCLIDEAN")

icd_pipelineModel = PipelineModel(
    stages = [
        documentAssembler,
        sbert_embedder,
        icd_resolver])

res = icd_pipelineModel.transform(spark.createDataFrame([["82 - year-old male with a history of hypertension , chronic renal insufficiency , COPD , and gastritis"]]).toDF("text"))
```

*Results* :
```bash
|    | chunk                       | entity  | code_3char | code_desc_3char                               | code_full | code_full_description                     |  distance | all_k_resolutions_3char                                                    | all_k_codes_3char                              |
|---:|:----------------------------|:--------|:-----------|:----------------------------------------------|:----------|:------------------------------------------|----------:|:---------------------------------------------------------------------------|:-----------------------------------------------|
|  0 | hypertension                | SYMPTOM | I10        | hypertension                                  | I150      | Renovascular hypertension                 |    0      | [hypertension, hypertension (high blood pressure), h/o: hypertension, ...] | [I10, I15, Z86, Z82, I11, R03, Z87, E27]       |
|  1 | chronic renal insufficiency | SYMPTOM | N18        | chronic renal impairment                      | N186      | End stage renal disease                   |    0.014  | [chronic renal impairment, renal insufficiency, renal failure, anaemi ...] | [N18, P96, N19, D63, N28, Z87, N17, N25, R94]  |
|  2 | COPD                        | SYMPTOM | J44        | chronic obstructive lung disease (disorder)   | I2781     | Cor pulmonale (chronic)                   |    0.1197 | [chronic obstructive lung disease (disorder), chronic obstructive pul ...] | [J44, Z76, J81, J96, R06, I27, Z87]            |
|  3 | gastritis                   | SYMPTOM | K29        | gastritis                                     | K5281     | Eosinophilic gastritis or gastroenteritis |    0      | gastritis:::bacterial gastritis:::parasitic gastritis                      | [K29, B96, K93]                                |
```

+ `sbiobertresolve_rxnorm_ndc` : This model maps `DRUG` entities to rxnorm codes and their [National Drug Codes (NDC)](https://www.drugs.com/ndc.html#:~:text=The%20NDC%2C%20or%20National%20Drug,and%20the%20commercial%20package%20size.)
 using `sbiobert_base_cased_mli` sentence embeddings. You can find all NDC codes of drugs seperated by `|` in the `all_k_aux_labels` parameter of the metadata.

*Example* :

```bash
documentAssembler = DocumentAssembler()\
      .setInputCol("text")\
      .setOutputCol("ner_chunk")

sbert_embedder = BertSentenceEmbeddings.pretrained('sbiobert_base_cased_mli', 'en','clinical/models')\
      .setInputCols(["ner_chunk"])\
      .setOutputCol("sentence_embeddings")

rxnorm_ndc_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_rxnorm_ndc", "en", "clinical/models") \
      .setInputCols(["ner_chunk", "sentence_embeddings"]) \
      .setOutputCol("rxnorm_code")\
      .setDistanceFunction("EUCLIDEAN")

rxnorm_ndc_pipelineModel = PipelineModel(
    stages = [
        documentAssembler,
        sbert_embedder,
        rxnorm_ndc_resolver])

res = rxnorm_ndc_pipelineModel.transform(spark.createDataFrame([["activated charcoal 30000 mg powder for oral suspension"]]).toDF("text"))
```

*Results* :

|chunk|rxnorm_code|all_codes|resolutions|all_k_aux_labels|all_distances|
|-|-|-|-|-|-|
|activated charcoal 30000 mg powder for oral suspension|1440919|1440919, 808917, 1088194, 1191772, 808921,...|activated charcoal 30000 MG Powder for Oral Suspension, Activated Charcoal 30000 MG Powder for Oral Suspension, wheat dextrin 3000 MG Powder for Oral Solution [Benefiber], cellulose 3000 MG Oral Powder [Unifiber], fosfomycin 3000 MG Powder for Oral Solution [Monurol] ...|69784030828, 00395052791, 08679001362\|86790016280\|00067004490, 46017004408\|68220004416, 00456430001,...|0.0000, 0.0000, 0.1128, 0.1148, 0.1201,...|

#### Updated UMLS Entity Resolvers (Dropping Invalid Codes)

UMLS model `sbiobertresolve_umls_findings` and `sbiobertresolve_umls_major_concepts` were updated by dropping the invalid codes using the [latest UMLS release](
https://www.nlm.nih.gov/pubs/techbull/mj21/mj21_umls_2021aa_release.html) done May 2021.  

#### 5 New Clinical NER Models (Trained By BertForTokenClassification Approach)

We are releasing four new BERT-based NER models.

+ `bert_token_classifier_ner_ade` : This model is BERT-Based version of `ner_ade_clinical` model and performs 5% better. It can detect drugs and adverse reactions of drugs in reviews, tweets, and medical texts using `DRUG` and `ADE` labels.

*Example* :

```bash
...
tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_ner_ade", "en", "clinical/models")\
    .setInputCols("token", "document")\
    .setOutputCol("ner")\
    .setCaseSensitive(True)

ner_converter = NerConverter()\
        .setInputCols(["document","token","ner"])\
        .setOutputCol("ner_chunk")

pipeline =  Pipeline(stages=[documentAssembler, tokenizer, tokenClassifier, ner_converter])
p_model = pipeline.fit(spark.createDataFrame(pd.DataFrame({'text': ['']})))

test_sentence = """Been taking Lipitor for 15 years , have experienced severe fatigue a lot!!! . Doctor moved me to voltaren 2 months ago , so far , have only experienced cramps"""
result = p_model.transform(spark.createDataFrame(pd.DataFrame({'text': [test_sentence]})))
```

*Results* :

```bash
+--------------+---------+
|chunk         |ner_label|
+--------------+---------+
|Lipitor       |DRUG     |
|severe fatigue|ADE      |
|voltaren      |DRUG     |
|cramps        |ADE      |
+--------------+---------+
```

+ `bert_token_classifier_ner_jsl_slim` : This model is BERT-Based version of `ner_jsl_slim` model and 2% better than the legacy NER model (MedicalNerModel) that is based on BiLSTM-CNN-Char architecture. It can detect `Death_Entity`, `Medical_Device`, `Vital_Sign`, `Alergen`, `Drug`, `Clinical_Dept`, `Lifestyle`, `Symptom`, `Body_Part`, `Physical_Measurement`, `Admission_Discharge`, `Date_Time`, `Age`, `Birth_Entity`, `Header`, `Oncological`, `Substance_Quantity`, `Test_Result`, `Test`, `Procedure`, `Treatment`, `Disease_Syndrome_Disorder`, `Pregnancy_Newborn`, `Demographics` entities.

*Example* :

```bash
...
tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_ner_jsl_slim", "en", "clinical/models")\
    .setInputCols("token", "document")\
    .setOutputCol("ner")\
    .setCaseSensitive(True)

ner_converter = NerConverter()\
    .setInputCols(["sentence","token","ner"])\
    .setOutputCol("ner_chunk")

pipeline = Pipeline(stages=[documentAssembler, sentence_detector, tokenizer, tokenClassifier, ner_converter])
p_model = pipeline.fit(spark.createDataFrame(pd.DataFrame({'text': ['']})))

test_sentence = """HISTORY: 30-year-old female presents for digital bilateral mammography secondary to a soft tissue lump palpated by the patient in the upper right shoulder. The patient has a family history of breast cancer within her mother at age 58. Patient denies personal history of breast cancer."""
result = p_model.transform(spark.createDataFrame(pd.DataFrame({'text': [test_sentence]})))
```

*Results* :

```bash
+----------------+------------+
|chunk           |ner_label   |
+----------------+------------+
|HISTORY:        |Header      |
|30-year-old     |Age         |
|female          |Demographics|
|mammography     |Test        |
|soft tissue lump|Symptom     |
|shoulder        |Body_Part   |
|breast cancer   |Oncological |
|her mother      |Demographics|
|age 58          |Age         |
|breast cancer   |Oncological |
+----------------+------------+
```

+ `bert_token_classifier_ner_drugs` : This model is BERT-based version of `ner_drugs` model and detects drug chemicals. This new model is 3% better than the legacy NER model (MedicalNerModel) that is based on BiLSTM-CNN-Char architecture.


*Example* :

```bash
...
tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_ner_drugs", "en", "clinical/models")\
  .setInputCols("token", "sentence")\
  .setOutputCol("ner")\
  .setCaseSensitive(True)

ner_converter = NerConverter()\
        .setInputCols(["sentence","token","ner"])\
        .setOutputCol("ner_chunk")

pipeline =  Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, tokenClassifier, ner_converter])
model = pipeline.fit(spark.createDataFrame(pd.DataFrame({'text': ['']})))

test_sentence = """The human KCNJ9 (Kir 3.3, GIRK3) is a member of the G-protein-activated inwardly rectifying potassium (GIRK) channel family. Here we describe the genomicorganization of the KCNJ9 locus on chromosome 1q21-23 as a candidate gene forType II diabetes mellitus in the Pima Indian population. The gene spansapproximately 7.6 kb and contains one noncoding and two coding exons separated byapproximately 2.2 and approximately 2.6 kb introns, respectively. We identified14 single nucleotide polymorphisms (SNPs), including one that predicts aVal366Ala substitution, and an 8 base-pair (bp) insertion/deletion. Ourexpression studies revealed the presence of the transcript in various humantissues including pancreas, and two major insulin-responsive tissues: fat andskeletal muscle. The characterization of the KCNJ9 gene should facilitate furtherstudies on the function of the KCNJ9 protein and allow evaluation of thepotential role of the locus in Type II diabetes.BACKGROUND: At present, it is one of the most important issues for the treatment of breast cancer to develop the standard therapy for patients previously treated with anthracyclines and taxanes. With the objective of determining the usefulnessof vinorelbine monotherapy in patients with advanced or recurrent breast cancerafter standard therapy, we evaluated the efficacy and safety of vinorelbine inpatients previously treated with anthracyclines and taxanes."""
result = model.transform(spark.createDataFrame(pd.DataFrame({'text': [test_sentence]})))
```
*Results* :

```bash
+--------------+---------+
|chunk         |ner_label|
+--------------+---------+
|potassium     |DrugChem |
|nucleotide    |DrugChem |
|anthracyclines|DrugChem |
|taxanes       |DrugChem |
|vinorelbine   |DrugChem |
|vinorelbine   |DrugChem |
|anthracyclines|DrugChem |
|taxanes       |DrugChem |
+--------------+---------+
```

+ `bert_token_classifier_ner_anatomy` : This model is BERT-Based version of `ner_anatomy` model and 3% better. It can detect `Anatomical_system`, `Cell`, `Cellular_component`, `Developing_anatomical_structure`, `Immaterial_anatomical_entity`, `Multi-tissue_structure`, `Organ`, `Organism_subdivision`, `Organism_substance`, `Pathological_formation`, `Tissue` entities.

*Example* :

```bash
...
tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_ner_anatomy", "en", "clinical/models")\
    .setInputCols("token", "sentence")\
    .setOutputCol("ner")\
    .setCaseSensitive(True)

ner_converter = NerConverter()\
    .setInputCols(["sentence","token","ner"])\
    .setOutputCol("ner_chunk")

pipeline =  Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, tokenClassifier, ner_converter])
pp_model = pipeline.fit(spark.createDataFrame(pd.DataFrame({'text': ['']})))

test_sentence = """This is an 11-year-old female who comes in for two different things. 1. She was seen by the allergist. No allergies present, so she stopped her Allegra, but she is still real congested and does a lot of snorting. They do not notice a lot of snoring at night though, but she seems to be always like that. 2. On her right great toe, she has got some redness and erythema. Her skin is kind of peeling a little bit, but it has been like that for about a week and a half now.\nGeneral: Well-developed female, in no acute distress, afebrile.\nHEENT: Sclerae and conjunctivae clear. Extraocular muscles intact. TMs clear. Nares patent. A little bit of swelling of the turbinates on the left. Oropharynx is essentially clear. Mucous membranes are moist.\nNeck: No lymphadenopathy.\nChest: Clear.\nAbdomen: Positive bowel sounds and soft.\nDermatologic: She has got redness along her right great toe, but no bleeding or oozing. Some dryness of her skin. Her toenails themselves are very short and even on her left foot and her left great toe the toenails are very short."""
result = pp_model.transform(spark.createDataFrame(pd.DataFrame({'text': [test_sentence]})))
```

*Results* :

```bash
+-------------------+----------------------+
|chunk              |ner_label             |
+-------------------+----------------------+
|great toe          |Multi-tissue_structure|
|skin               |Organ                 |
|conjunctivae       |Multi-tissue_structure|
|Extraocular muscles|Multi-tissue_structure|
|Nares              |Multi-tissue_structure|
|turbinates         |Multi-tissue_structure|
|Oropharynx         |Multi-tissue_structure|
|Mucous membranes   |Tissue                |
|Neck               |Organism_subdivision  |
|bowel              |Organ                 |
|great toe          |Multi-tissue_structure|
|skin               |Organ                 |
|toenails           |Organism_subdivision  |
|foot               |Organism_subdivision  |
|great toe          |Multi-tissue_structure|
|toenails           |Organism_subdivision  |
+-------------------+----------------------+
```

+ `bert_token_classifier_ner_bacteria` : This model is BERT-Based version of `ner_bacterial_species` model and detects different types of species of bacteria in clinical texts using `SPECIES` label.

*Example* :

```bash
...
tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_ner_bacteria", "en", "clinical/models")\
    .setInputCols("token", "document")\
    .setOutputCol("ner")\
    .setCaseSensitive(True)

ner_converter = NerConverter()\
    .setInputCols(["document","token","ner"])\
    .setOutputCol("ner_chunk")

pipeline =  Pipeline(stages=[documentAssembler, tokenizer, tokenClassifier, ner_converter])
p_model = pipeline.fit(spark.createDataFrame(pd.DataFrame({'text': ['']})))

test_sentence = """Based on these genetic and phenotypic properties, we propose that strain SMSP (T) represents \
a novel species of the genus Methanoregula, for which we propose the name Methanoregula formicica \
sp. nov., with the type strain SMSP (T) (= NBRC 105244 (T) = DSM 22288 (T))."""
result = p_model.transform(spark.createDataFrame(pd.DataFrame({'text': [test_sentence]})))
```

*Results* :

```bash
+-----------------------+---------+
|chunk                  |ner_label|
+-----------------------+---------+
|SMSP (T)               |SPECIES  |
|Methanoregula formicica|SPECIES  |
|SMSP (T)               |SPECIES  |
+-----------------------+---------+
```

#### Radiology NER Model Trained On cheXpert Dataset

+ Ner NER model `ner_chexpert` trained on Radiology Chest reports to extract anatomical sites and observation entities. The model achieves 92.8% and 77.4% micro and macro f1 scores on the cheXpert dataset.

*Example* :

```bash
...
embeddings_clinical = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")  .setInputCols(["sentence", "token"])  .setOutputCol("embeddings")
clinical_ner = MedicalNerModel.pretrained("ner_chexpert", "en", "clinical/models")   .setInputCols(["sentence", "token", "embeddings"])   .setOutputCol("ner")
...
nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings_clinical, clinical_ner, ner_converter])
model = nlpPipeline.fit(spark.createDataFrame([[""]]).toDF("text"))
EXAMPLE_TEXT = """FINAL REPORT HISTORY : Chest tube leak , to assess for pneumothorax .
FINDINGS : In comparison with study of ___ , the endotracheal tube and Swan - Ganz catheter have been removed . The left chest tube remains in place and there is no evidence of pneumothorax. Mild atelectatic changes are seen at the left base."""
results = model.transform(spark.createDataFrame([[EXAMPLE_TEXT]]).toDF("text"))
```

*Results* :

```bash
|    | chunk                    | label   |
|---:|:-------------------------|:--------|
|  0 | endotracheal tube        | OBS     |
|  1 | Swan - Ganz catheter     | OBS     |
|  2 | left chest               | ANAT    |
|  3 | tube                     | OBS     |
|  4 | in place                 | OBS     |
|  5 | pneumothorax             | OBS     |
|  6 | Mild atelectatic changes | OBS     |
|  7 | left base                | ANAT    |
```

#### New Speed Benchmarks on Databricks

We prepared a speed benchmark table by running a NER pipeline on various number of cluster configurations (worker number, driver node, specs etc) and also writing the results to parquet or delta formats. You can find all the details of these tries in here : [Speed Benchmark Table](https://nlp.johnsnowlabs.com/docs/en/benchmark)

#### NerConverterInternal Fixes
Now NerConverterInternal can deal with tags that have some dash (`-`) charachter like B-GENE-N and B-GENE-Y.


#### Simplified Setup and Recommended Use of start() Function
Starting with this release, we are shipping AWS credentials inside Spark NLP Healthcare's license. This removes the requirement of setting the `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` environment variables.
To use this feature, you just need to make sure that you always call the start() function at the beginning of your program,

```python
from sparknlp_jsl import start
spark = start()
```

```scala
import com.johnsnowlabs.util.start
val spark = start()
```

If for some reason you don't want to use this mechanism, the keys will continue to be shipped separately, and the environment variables will continue to work as they did in the past.

#### Ner Evaluation Metrics Fix

Bug fixed in the `NerDLMetrics` package. Previously, the `full_chunk` option was using greedy approach to merge chunks for a strict evaluation, which has been fixed to merge chunks using IOB scheme to get accurate entities boundaries and metrics. Also, the `tag` option has been fixed to get metrics that align with the default NER logs.

#### New Notebooks

- [Clinical Relation Extraction Knowledge Graph with Neo4j Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/10.2.Clinical_RE_Knowledge_Graph_with_Neo4j.ipynb)
- [NER Profiling Pretrained Pipelines Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/11.2.Pretrained_NER_Profiling_Pipelines.ipynb)
- New Databricks [Detecting Adverse Drug Events From Conversational Texts](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/databricks/python/healthcare_case_studies/Detecting%20Adverse%20Drug%20Events%20From%20Conversational%20Texts.ipynb) case study notebook.

**To see more, please check :** [Spark NLP Healthcare Workshop Repo](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings/Healthcare)

<div class="prev_ver h3-box" markdown="1">

## Versions

</div>

<ul class="pagination">
    <li>
        <a href="release_notes_3_2_3">Version 3.2.3</a>
    </li>
    <li>
        <strong>Version 3.3.0</strong>
    </li>
    <li>
        <a href="release_notes_3_3_1">Version 3.3.1</a>
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
    <li><a href="release_notes_3_4_0">3.4.0</a></li>
    <li><a href="release_notes_3_3_4">3.3.4</a></li>
    <li><a href="release_notes_3_3_2">3.3.2</a></li>
    <li><a href="release_notes_3_3_1">3.3.1</a></li>
    <li class="active"><a href="release_notes_3_3_0">3.3.0</a></li>
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