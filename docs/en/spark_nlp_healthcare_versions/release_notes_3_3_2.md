---
layout: docs
header: true
seotitle: Spark NLP for Healthcare | John Snow Labs
title: Spark NLP for Healthcare Release Notes 3.3.2
permalink: /docs/en/spark_nlp_healthcare_versions/release_notes_3_3_2
key: docs-licensed-release-notes
modify_date: 2021-07-14
show_nav: true
sidebar:
    nav: sparknlp-healthcare
---

## 3.3.2
We are glad to announce that Spark NLP Healthcare 3.3.2 has been released!.



#### Highlights

+ New Clinical NER Models and Spanish NER Model
+ New BERT-Based Clinical NER Models
+ Updated Clinical NER Model
+ New NER Model Class Distribution Feature
+ New RxNorm Sentence Entity Resolver Model
+ New Spanish SNOMED Sentence Entity Resolver Model
+ New Clinical Question vs Statement BertForSequenceClassification model
+ New Sentence Entity Resolver Fine-Tune Features (Overwriting and Drop Code)
+ Updated ICD10CM Entity Resolver Models
+ Updated NER Profiling Pretrained Pipelines
+ New ChunkSentenceSplitter Annotator
+ Updated Spark NLP For Healthcare Notebooks and New Notebooks

#### New Clinical NER Models (including a new Spanish one)

We are releasing three new clinical NER models trained by MedicalNerApproach().

+ `roberta_ner_diag_proc` : This models leverages Spanish Roberta Biomedical Embeddings (`roberta_base_biomedical`) to extract two entities, Diagnosis and Procedures (`DIAGNOSTICO`, `PROCEDIMIENTO`). It's a renewed version of `ner_diag_proc_es`, available [here](https://nlp.johnsnowlabs.com/2020/07/08/ner_diag_proc_es.html), that was trained with `embeddings_scielowiki_300d` embeddings instead.

*Example* :

```bash
...
embeddings =  RoBertaEmbeddings.pretrained("roberta_base_biomedical", "es")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")

ner = MedicalNerModel.pretrained("roberta_ner_diag_proc", "es", "clinical/models")\
    .setInputCols(["sentence", "token", "embeddings"])\
    .setOutputCol("ner")\

ner_converter = NerConverter() \
    .setInputCols(['sentence', 'token', 'ner']) \
    .setOutputCol('ner_chunk')

pipeline = Pipeline(stages = [
    documentAssembler,
    sentenceDetector,
    tokenizer,
    embeddings,
    ner,
    ner_converter])

empty = spark.createDataFrame([['']]).toDF("text")

p_model = pipeline.fit(empty)

test_sentence = 'Mujer de 28 años con antecedentes de diabetes mellitus gestacional diagnosticada ocho años antes de la presentación y posterior diabetes mellitus tipo dos (DM2), un episodio previo de pancreatitis inducida por HTG tres años antes de la presentación, asociado con una hepatitis aguda, y obesidad con un índice de masa corporal (IMC) de 33,5 kg / m2, que se presentó con antecedentes de una semana de poliuria, polidipsia, falta de apetito y vómitos. Dos semanas antes de la presentación, fue tratada con un ciclo de cinco días de amoxicilina por una infección del tracto respiratorio. Estaba tomando metformina, glipizida y dapagliflozina para la DM2 y atorvastatina y gemfibrozil para la HTG. Había estado tomando dapagliflozina durante seis meses en el momento de la presentación. El examen físico al momento de la presentación fue significativo para la mucosa oral seca; significativamente, su examen abdominal fue benigno sin dolor a la palpación, protección o rigidez. Los hallazgos de laboratorio pertinentes al ingreso fueron: glucosa sérica 111 mg / dl, bicarbonato 18 mmol / l, anión gap 20, creatinina 0,4 mg / dl, triglicéridos 508 mg / dl, colesterol total 122 mg / dl, hemoglobina glucosilada (HbA1c) 10%. y pH venoso 7,27. La lipasa sérica fue normal a 43 U / L. Los niveles séricos de acetona no pudieron evaluarse ya que las muestras de sangre se mantuvieron hemolizadas debido a una lipemia significativa. La paciente ingresó inicialmente por cetosis por inanición, ya que refirió una ingesta oral deficiente durante los tres días Previous a la admisión. Sin embargo, la química sérica obtenida seis horas después de la presentación reveló que su glucosa era de 186 mg / dL, la brecha aniónica todavía estaba elevada a 21, el bicarbonato sérico era de 16 mmol / L, el nivel de triglicéridos alcanzó un máximo de 2050 mg / dL y la lipasa fue de 52 U / L. Se obtuvo el nivel de β-hidroxibutirato y se encontró que estaba elevado a 5,29 mmol / L; la muestra original se centrifugó y la capa de quilomicrones se eliminó antes del análisis debido a la interferencia de la turbidez causada por la lipemia nuevamente. El paciente fue tratado con un goteo de insulina para euDKA y HTG con una reducción de la brecha aniónica a 13 y triglicéridos a 1400 mg / dL, dentro de las 24 horas. Se pensó que su euDKA fue precipitada por su infección del tracto respiratorio en el contexto del uso del inhibidor de SGLT2. La paciente fue atendida por el servicio de endocrinología y fue dada de alta con 40 unidades de insulina glargina por la noche, 12 unidades de insulina lispro con las comidas y metformina 1000 mg dos veces al día. Se determinó que todos los inhibidores de SGLT2 deben suspenderse indefinidamente. Tuvo un seguimiento estrecho con endocrinología post alta.'
res = p_model.transform(spark.createDataFrame(pd.DataFrame({'text': [test_sentence]})))
```

*Results* :
```bash
+---------------------------------+------------+
|                             text|ner_label  |
+---------------------------------+------------+
|    diabetes mellitus gestacional|DIAGNOSTICO|
|       diabetes mellitus tipo dos|DIAGNOSTICO|
|                              DM2|DIAGNOSTICO|
|    pancreatitis inducida por HTG|DIAGNOSTICO|
|                  hepatitis aguda|DIAGNOSTICO|
|                         obesidad|DIAGNOSTICO|
|          índice de masa corporal|DIAGNOSTICO|
|                              IMC|DIAGNOSTICO|
|                         poliuria|DIAGNOSTICO|
|                       polidipsia|DIAGNOSTICO|
|                          vómitos|DIAGNOSTICO|
|infección del tracto respiratorio|DIAGNOSTICO|
|                              DM2|DIAGNOSTICO|
|                              HTG|DIAGNOSTICO|
|                            dolor|DIAGNOSTICO|
|                          rigidez|DIAGNOSTICO|
|                          cetosis|DIAGNOSTICO|
|infección del tracto respiratorio|DIAGNOSTICO|
+---------------------------------+-----------+
```

+ `ner_covid_trials` : This model is trained to extract covid-specific medical entities in clinical trials. It supports the following entities ranging from virus type to trial design: `Stage`, `Severity`, `Virus`, `Trial_Design`, `Trial_Phase`, `N_Patients`, `Institution`, `Statistical_Indicator`, `Section_Header`, `Cell_Type`, `Cellular_component`, `Viral_components`, `Physiological_reaction`, `Biological_molecules`, `Admission_Discharge`, `Age`, `BMI`, `Cerebrovascular_Disease`, `Date`, `Death_Entity`, `Diabetes`, `Disease_Syndrome_Disorder`, `Dosage`, `Drug_Ingredient`, `Employment`, `Frequency`, `Gender`, `Heart_Disease`, `Hypertension`, `Obesity`, `Pulse`, `Race_Ethnicity`, `Respiration`, `Route`, `Smoking`, `Time`, `Total_Cholesterol`, `Treatment`, `VS_Finding`, `Vaccine` .

*Example* :
```bash
...
covid_ner = MedicalNerModel.pretrained('ner_covid_trials', 'en', 'clinical/models') \
      .setInputCols(["sentence", "token", "embeddings"]) \
      .setOutputCol("ner")    
...

results = covid_model.transform(spark.createDataFrame(pd.DataFrame({"text": ["""In December 2019 , a group of patients with the acute respiratory disease was detected in Wuhan , Hubei Province of China . A month later , a new beta-coronavirus was identified as the cause of the 2019 coronavirus infection . SARS-CoV-2 is a coronavirus that belongs to the group of β-coronaviruses of the subgenus Coronaviridae . The SARS-CoV-2 is the third known zoonotic coronavirus disease after severe acute respiratory syndrome ( SARS ) and Middle Eastern respiratory syndrome ( MERS ). The diagnosis of SARS-CoV-2 recommended by the WHO , CDC is the collection of a sample from the upper respiratory tract ( nasal and oropharyngeal exudate ) or from the lower respiratory tract such as expectoration of endotracheal aspirate and bronchioloalveolar lavage and its analysis using the test of real-time polymerase chain reaction ( qRT-PCR )."""]})))
```

*Results* :

```bash

|    | chunk                               |   begin |   end | entity                    |
|---:|:------------------------------------|--------:|------:|:--------------------------|
|  0 | December 2019                       |       3 |    15 | Date                      |
|  1 | acute respiratory disease           |      48 |    72 | Disease_Syndrome_Disorder |
|  2 | beta-coronavirus                    |     146 |   161 | Virus                     |
|  3 | 2019 coronavirus infection          |     198 |   223 | Disease_Syndrome_Disorder |
|  4 | SARS-CoV-2                          |     227 |   236 | Virus                     |
|  5 | coronavirus                         |     243 |   253 | Virus                     |
|  6 | β-coronaviruses                     |     284 |   298 | Virus                     |
|  7 | subgenus Coronaviridae              |     307 |   328 | Virus                     |
|  8 | SARS-CoV-2                          |     336 |   345 | Virus                     |
|  9 | zoonotic coronavirus disease        |     366 |   393 | Disease_Syndrome_Disorder |
| 10 | severe acute respiratory syndrome   |     401 |   433 | Disease_Syndrome_Disorder |
| 11 | SARS                                |     437 |   440 | Disease_Syndrome_Disorder |
| 12 | Middle Eastern respiratory syndrome |     448 |   482 | Disease_Syndrome_Disorder |
| 13 | MERS                                |     486 |   489 | Disease_Syndrome_Disorder |
| 14 | SARS-CoV-2                          |     511 |   520 | Virus                     |
| 15 | WHO                                 |     541 |   543 | Institution               |
| 16 | CDC                                 |     547 |   549 | Institution               |
```

+ `ner_chemd_clinical` : This model extract the names of chemical compounds and drugs in medical texts. The entities that can be detected are as follows : `SYSTEMATIC`, `IDENTIFIERS`, `FORMULA`, `TRIVIAL`, `ABBREVIATION`, `FAMILY`, `MULTIPLE` . For reference [click here](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4331685/) .

*Example* :
```bash
...
chemd_ner = MedicalNerModel.pretrained('ner_chemd', 'en', 'clinical/models') \
      .setInputCols(["sentence", "token", "embeddings"]) \
      .setOutputCol("ner")    
...

results = chemd_model.transform(spark.createDataFrame(pd.DataFrame({"text": ["""Isolation, Structure Elucidation, and Iron-Binding Properties of Lystabactins, Siderophores Isolated from a Marine Pseudoalteromonas sp. The marine bacterium Pseudoalteromonas sp. S2B, isolated from the Gulf of Mexico after the Deepwater Horizon oil spill, was found to produce lystabactins A, B, and C (1-3), three new siderophores. The structures were elucidated through mass spectrometry, amino acid analysis, and NMR. The lystabactins are composed of serine (Ser), asparagine (Asn), two formylated/hydroxylated ornithines (FOHOrn), dihydroxy benzoic acid (Dhb), and a very unusual nonproteinogenic amino acid, 4,8-diamino-3-hydroxyoctanoic acid (LySta). The iron-binding properties of the compounds were investigated through a spectrophotometric competition."""]})))
```

*Results* :

```bash
+----------------------------------+------------+
|chunk                             |ner_label   |
+----------------------------------+------------+
|Lystabactins                      |FAMILY      |
|lystabactins A, B, and C          |MULTIPLE    |
|amino acid                        |FAMILY      |
|lystabactins                      |FAMILY      |
|serine                            |TRIVIAL     |
|Ser                               |FORMULA     |
|asparagine                        |TRIVIAL     |
|Asn                               |FORMULA     |
|formylated/hydroxylated ornithines|FAMILY      |
|FOHOrn                            |FORMULA     |
|dihydroxy benzoic acid            |SYSTEMATIC  |
|amino acid                        |FAMILY      |
|4,8-diamino-3-hydroxyoctanoic acid|SYSTEMATIC  |
|LySta                             |ABBREVIATION|
+----------------------------------+------------+
```

#### New BERT-Based Clinical NER Models

We have two new BERT-Based token classifier NER models.

+ `bert_token_classifier_ner_bionlp` : This model is BERT-based version of `ner_bionlp` model and can detect biological and genetics terms in cancer-related texts. (`Amino_acid`, `Anatomical_system`, `Cancer`, `Cell`, `Cellular_component`, `Developing_anatomical_Structure`, `Gene_or_gene_product`, `Immaterial_anatomical_entity`, `Multi-tissue_structure`, `Organ`, `Organism`, `Organism_subdivision`, `Simple_chemical`, `Tissue`)

*Example* :

```python
...
tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_ner_bionlp", "en", "clinical/models")\
    .setInputCols("token", "document")\
    .setOutputCol("ner")\
    .setCaseSensitive(True)
...

test_sentence = """Both the erbA IRES and the erbA/myb virus constructs transformed erythroid cells after infection of bone marrow or blastoderm cultures. The erbA/myb IRES virus exhibited a 5-10-fold higher transformed colony forming efficiency than the erbA IRES virus in the blastoderm assay."""
result = p_model.transform(spark.createDataFrame(pd.DataFrame({'text': [test_sentence]})))
```

*Results* :

```bash
+-------------------+----------------------+
|chunk              |ner_label             |
+-------------------+----------------------+
|erbA IRES          |Organism              |
|erbA/myb virus     |Organism              |
|erythroid cells    |Cell                  |
|bone marrow        |Multi-tissue_structure|
|blastoderm cultures|Cell                  |
|erbA/myb IRES virus|Organism              |
|erbA IRES virus    |Organism              |
|blastoderm         |Cell                  |
+-------------------+----------------------+
```

+ `bert_token_classifier_ner_cellular` : This model is BERT-based version of `ner_cellular` model and can detect molecular biology-related terms (`DNA`, `Cell_type`, `Cell_line`, `RNA`, `Protein`) in medical texts.

*Metrics* :

```bash
              precision    recall  f1-score   support

       B-DNA       0.87      0.77      0.82      1056
       B-RNA       0.85      0.79      0.82       118
 B-cell_line       0.66      0.70      0.68       500
 B-cell_type       0.87      0.75      0.81      1921
   B-protein       0.90      0.85      0.88      5067
       I-DNA       0.93      0.86      0.90      1789
       I-RNA       0.92      0.84      0.88       187
 I-cell_line       0.67      0.76      0.71       989
 I-cell_type       0.92      0.76      0.84      2991
   I-protein       0.94      0.80      0.87      4774

    accuracy                           0.80     19392
   macro avg       0.76      0.81      0.78     19392
weighted avg       0.89      0.80      0.85     19392
```

*Example* :

```python
...

tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_ner_cellular", "en", "clinical/models")
.setInputCols("token", "document")
.setOutputCol("ner")
.setCaseSensitive(True)

...

test_sentence = """Detection of various other intracellular signaling proteins is also described. Genetic characterization of transactivation of the human T-cell leukemia virus type 1 promoter: Binding of Tax to Tax-responsive element 1 is mediated by the cyclic AMP-responsive members of the CREB/ATF family of transcription factors. To achieve a better understanding of the mechanism of transactivation by Tax of human T-cell leukemia virus type 1 Tax-responsive element 1 (TRE-1), we developed a genetic approach with Saccharomyces cerevisiae. We constructed a yeast reporter strain containing the lacZ gene under the control of the CYC1 promoter associated with three copies of TRE-1. Expression of either the cyclic AMP response element-binding protein (CREB) or CREB fused to the GAL4 activation domain (GAD) in this strain did not modify the expression of the reporter gene. Tax alone was also inactive."""

result = p_model.transform(spark.createDataFrame(pd.DataFrame({'text': [test_sentence]})))
```

*Results* :

```bash
+-------------------------------------------+---------+
|chunk                                      |ner_label|
+-------------------------------------------+---------+
|intracellular signaling proteins           |protein  |
|human T-cell leukemia virus type 1 promoter|DNA      |
|Tax                                        |protein  |
|Tax-responsive element 1                   |DNA      |
|cyclic AMP-responsive members              |protein  |
|CREB/ATF family                            |protein  |
|transcription factors                      |protein  |
|Tax                                        |protein  |
|human T-cell leukemia virus type 1         |DNA      |
|Tax-responsive element 1                   |DNA      |
|TRE-1                                      |DNA      |
|lacZ gene                                  |DNA      |
|CYC1 promoter                              |DNA      |
|TRE-1                                      |DNA      |
|cyclic AMP response element-binding protein|protein  |
|CREB                                       |protein  |
|CREB                                       |protein  |
|GAL4 activation domain                     |protein  |
|GAD                                        |protein  |
|reporter gene                              |DNA      |
|Tax                                        |protein  |
+-------------------------------------------+---------+
```

#### Updated Clinical NER Model

We have updated `ner_jsl_enriched` model by enriching the training data using clinical trials data to make it more robust. This model is capable of predicting up to `87` different entities and is based on `ner_jsl` model. Here are the entities this model can detect;

`Social_History_Header`, `Oncology_Therapy`, `Blood_Pressure`, `Respiration`, `Performance_Status`, `Family_History_Header`, `Dosage`, `Clinical_Dept`, `Diet`, `Procedure`, `HDL`, `Weight`, `Admission_Discharge`, `LDL`, `Kidney_Disease`, `Oncological`, `Route`, `Imaging_Technique`, `Puerperium`, `Overweight`, `Temperature`, `Diabetes`, `Vaccine`, `Age`, `Test_Result`, `Employment`, `Time`, `Obesity`, `EKG_Findings`, `Pregnancy`, `Communicable_Disease`, `BMI`, `Strength`, `Tumor_Finding`, `Section_Header`, `RelativeDate`, `ImagingFindings`, `Death_Entity`, `Date`, `Cerebrovascular_Disease`, `Treatment`, `Labour_Delivery`, `Pregnancy_Delivery_Puerperium`, `Direction`, `Internal_organ_or_component`, `Psychological_Condition`, `Form`, `Medical_Device`, `Test`, `Symptom`, `Disease_Syndrome_Disorder`, `Staging`, `Birth_Entity`, `Hyperlipidemia`, `O2_Saturation`, `Frequency`, `External_body_part_or_region`, `Drug_Ingredient`, `Vital_Signs_Header`, `Substance_Quantity`, `Race_Ethnicity`, `VS_Finding`, `Injury_or_Poisoning`, `Medical_History_Header`, `Alcohol`, `Triglycerides`, `Total_Cholesterol`, `Sexually_Active_or_Sexual_Orientation`, `Female_Reproductive_Status`, `Relationship_Status`, `Drug_BrandName`, `RelativeTime`, `Duration`, `Hypertension`, `Metastasis`, `Gender`, `Oxygen_Therapy`, `Pulse`, `Heart_Disease`, `Modifier`, `Allergen`, `Smoking`, `Substance`, `Cancer_Modifier`, `Fetus_NewBorn`, `Height` .

*Example* :

```bash
...  
clinical_ner = MedicalNerModel.pretrained("ner_jsl_enriched", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner")
...

results = model.transform(spark.createDataFrame([["The patient is a 21-day-old Caucasian male here for 2 days of congestion - mom has been suctioning yellow discharge from the patient's nares, plus she has noticed some mild problems with his breathing while feeding (but negative for any perioral cyanosis or retractions). One day ago, mom also noticed a tactile temperature and gave the patient Tylenol. Baby also has had some decreased p.o. intake. His normal breast-feeding is down from 20 minutes q.2h. to 5 to 10 minutes secondary to his respiratory congestion. He sleeps well, but has been more tired and has been fussy over the past 2 days. The parents noticed no improvement with albuterol treatments given in the ER. His urine output has also decreased; normally he has 8 to 10 wet and 5 dirty diapers per 24 hours, now he has down to 4 wet diapers per 24 hours. Mom denies any diarrhea. His bowel movements are yellow colored and soft in nature."]], ["text"]))
```
*Results* :

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

#### New NER Model Class Distribution Feature

+ `getTrainingClassDistribution` : This parameter returns the distribution of labels used when training the NER model.

*Example*:

```bash
ner_model.getTrainingClassDistribution()
>> {'B-Disease': 2536, 'O': 31659, 'I-Disease': 2960}
```

#### New RxNorm Sentence Entity Resolver Model

+ `sbiobertresolve_rxnorm_augmented` : This model maps clinical entities and concepts (like drugs/ingredients) to RxNorm codes using sbiobert_base_cased_mli Sentence Bert Embeddings. It trained on the augmented version of the dataset which is used in previous RxNorm resolver models. Additionally, this model returns concept classes of the drugs in all_k_aux_labels column.

#### New Spanish SNOMED Sentence Entity Resolver Model

+ `robertaresolve_snomed` : This models leverages Spanish Roberta Biomedical Embeddings (`roberta_base_biomedical`) at sentence-level to map ner chunks into Spanish SNOMED codes.

*Example* :

```bash

documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentenceDetector = SentenceDetectorDLModel.pretrained() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence")

tokenizer = Tokenizer()\
    .setInputCols("sentence")\
    .setOutputCol("token")

word_embeddings = RoBertaEmbeddings.pretrained("roberta_base_biomedical", "es")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("roberta_embeddings")

ner = MedicalNerModel.pretrained("roberta_ner_diag_proc","es","clinical/models")\
    .setInputCols("sentence","token","roberta_embeddings")\
    .setOutputCol("ner")

ner_converter = NerConverter() \
    .setInputCols(["sentence", "token", "ner"]) \
    .setOutputCol("ner_chunk")

c2doc = Chunk2Doc() \
    .setInputCols(["ner_chunk"]) \
    .setOutputCol("ner_chunk_doc")

chunk_embeddings = SentenceEmbeddings() \
    .setInputCols(["ner_chunk_doc", "roberta_embeddings"]) \
    .setOutputCol("chunk_embeddings") \
    .setPoolingStrategy("AVERAGE")

er = SentenceEntityResolverModel.pretrained("robertaresolve_snomed", "es", "clinical/models")\
    .setInputCols(["ner_chunk_doc", "chunk_embeddings"]) \
    .setOutputCol("snomed_code") \
    .setDistanceFunction("EUCLIDEAN")

snomed_training_pipeline = Pipeline(stages = [
    documentAssembler,
    sentenceDetector,
    tokenizer,
    word_embeddings,
    ner,
    ner_converter,
    c2doc,
    chunk_embeddings,
    er])

empty = spark.createDataFrame([['']]).toDF("text")

p_model = snomed_pipeline .fit(empty)

test_sentence = 'Mujer de 28 años con antecedentes de diabetes mellitus gestacional diagnosticada ocho años antes de la presentación y posterior diabetes mellitus tipo dos (DM2), un episodio previo de pancreatitis inducida por HTG tres años antes de la presentación, asociado con una hepatitis aguda, y obesidad con un índice de masa corporal (IMC) de 33,5 kg / m2, que se presentó con antecedentes de una semana de poliuria, polidipsia, falta de apetito y vómitos. Dos semanas antes de la presentación, fue tratada con un ciclo de cinco días de amoxicilina por una infección del tracto respiratorio. Estaba tomando metformina, glipizida y dapagliflozina para la DM2 y atorvastatina y gemfibrozil para la HTG. Había estado tomando dapagliflozina durante seis meses en el momento de la presentación. El examen físico al momento de la presentación fue significativo para la mucosa oral seca; significativamente, su examen abdominal fue benigno sin dolor a la palpación, protección o rigidez. Los hallazgos de laboratorio pertinentes al ingreso fueron: glucosa sérica 111 mg / dl, bicarbonato 18 mmol / l, anión gap 20, creatinina 0,4 mg / dl, triglicéridos 508 mg / dl, colesterol total 122 mg / dl, hemoglobina glucosilada (HbA1c) 10%. y pH venoso 7,27. La lipasa sérica fue normal a 43 U / L. Los niveles séricos de acetona no pudieron evaluarse ya que las muestras de sangre se mantuvieron hemolizadas debido a una lipemia significativa. La paciente ingresó inicialmente por cetosis por inanición, ya que refirió una ingesta oral deficiente durante los tres días Previous a la admisión. Sin embargo, la química sérica obtenida seis horas después de la presentación reveló que su glucosa era de 186 mg / dL, la brecha aniónica todavía estaba elevada a 21, el bicarbonato sérico era de 16 mmol / L, el nivel de triglicéridos alcanzó un máximo de 2050 mg / dL y la lipasa fue de 52 U / L. Se obtuvo el nivel de β-hidroxibutirato y se encontró que estaba elevado a 5,29 mmol / L; la muestra original se centrifugó y la capa de quilomicrones se eliminó antes del análisis debido a la interferencia de la turbidez causada por la lipemia nuevamente. El paciente fue tratado con un goteo de insulina para euDKA y HTG con una reducción de la brecha aniónica a 13 y triglicéridos a 1400 mg / dL, dentro de las 24 horas. Se pensó que su euDKA fue precipitada por su infección del tracto respiratorio en el contexto del uso del inhibidor de SGLT2. La paciente fue atendida por el servicio de endocrinología y fue dada de alta con 40 unidades de insulina glargina por la noche, 12 unidades de insulina lispro con las comidas y metformina 1000 mg dos veces al día. Se determinó que todos los inhibidores de SGLT2 deben suspenderse indefinidamente. Tuvo un seguimiento estrecho con endocrinología post alta.'

res = p_model.transform(spark.createDataFrame(pd.DataFrame({'text': [test_sentence]})))
```

*Results* :

```bash
+----+-------------------------------+-------------+--------------+
|    | ner_chunk                     | entity      |   snomed_code|
|----+-------------------------------+-------------+--------------|
|  0 | diabetes mellitus gestacional | DIAGNOSTICO |     11687002 |
|  1 | diabetes mellitus tipo dos (  | DIAGNOSTICO |     44054006 |
|  2 | pancreatitis                  | DIAGNOSTICO |     75694006 |
|  3 | HTG                           | DIAGNOSTICO |    266569009 |
|  4 | hepatitis aguda               | DIAGNOSTICO |     37871000 |
|  5 | obesidad                      | DIAGNOSTICO |      5476005 |
|  6 | índice de masa corporal       | DIAGNOSTICO |    162859006 |
|  7 | poliuria                      | DIAGNOSTICO |     56574000 |
|  8 | polidipsia                    | DIAGNOSTICO |     17173007 |
|  9 | falta de apetito              | DIAGNOSTICO |     49233005 |
| 10 | vómitos                       | DIAGNOSTICO |    422400008 |
| 11 | infección                     | DIAGNOSTICO |     40733004 |
| 12 | HTG                           | DIAGNOSTICO |    266569009 |
| 13 | dolor                         | DIAGNOSTICO |     22253000 |
| 14 | rigidez                       | DIAGNOSTICO |    271587009 |
| 15 | cetosis                       | DIAGNOSTICO |      2538008 |
| 16 | infección                     | DIAGNOSTICO |     40733004 |
+----+-------------------------------+-------------+--------------+
```

#### New Clinical Question vs Statement BertForSequenceClassification model

+ `bert_sequence_classifier_question_statement_clinical` : This model classifies sentences into one of these two classes: question (interrogative sentence) or statement (declarative sentence) and trained with BertForSequenceClassification. This model is at first trained on SQuAD and SPAADIA dataset and then fine tuned on the clinical visit documents and MIMIC-III dataset annotated in-house. Using this model, you can find the question statements and exclude & utilize in the downstream tasks such as NER and relation extraction models.

*Example* :

```bash
documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentenceDetector = SentenceDetectorDLModel.pretrained() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence")

tokenizer = Tokenizer()\
    .setInputCols("sentence")\
    .setOutputCol("token")

seq = BertForSequenceClassification.pretrained('bert_sequence_classifier_question_statement_clinical', 'en', 'clinical/models')\
  .setInputCols(["token", "sentence"])\
  .setOutputCol("label")\
  .setCaseSensitive(True)

pipeline = Pipeline(stages = [
    documentAssembler,
    sentenceDetector,
    tokenizer,
    seq])

test_sentences = ["""Hello I am going to be having a baby throughand have just received my medical results before I have my tubes tested. I had the tests on day 23 of my cycle. My progresterone level is 10. What does this mean? What does progesterone level of 10 indicate?
Your progesterone report is perfectly normal. We expect this result on day 23rd of the cycle.So there's nothing to worry as it's perfectly alright"""]

res = p_model.transform(spark.createDataFrame(pd.DataFrame({'text': test_sentences})))
```

*Results* :

```bash
+--------------------------------------------------------------------------------------------------------------------+---------+
|sentence                                                                                                            |label    |
+--------------------------------------------------------------------------------------------------------------------+---------+
|Hello I am going to be having a baby throughand have just received my medical results before I have my tubes tested.|statement|
|I had the tests on day 23 of my cycle.                                                                              |statement|
|My progresterone level is 10.                                                                                       |statement|
|What does this mean?                                                                                                |question |
|What does progesterone level of 10 indicate?                                                                        |question |
|Your progesterone report is perfectly normal. We expect this result on day 23rd of the cycle.                       |statement|
|So there's nothing to worry as it's perfectly alright                                                               |statement|
+--------------------------------------------------------------------------------------------------------------------+---------
```

*Metrics* :
```bash
              precision    recall  f1-score   support

    question       0.97      0.94      0.96       243
   statement       0.98      0.99      0.99       729

    accuracy                           0.98       972
   macro avg       0.98      0.97      0.97       972
weighted avg       0.98      0.98      0.98       972
```

#### New Sentence Entity Resolver Fine-Tune Features (Overwriting and Drop Code)

+ `.setOverwriteExistingCode()` : This parameter provides overwriting codes over the existing codes if in pretrained Sentence Entity Resolver Model. For example, you want to add a new term to a pretrained resolver model, and if the code of term already exists in the pretrained model, when you `.setOverwriteExistingCode(True)`, it removes all the same codes and their descriptions from the model, then you will have just the new term with its code in the fine-tuned model.

+ `.setDropCodesList()` : This parameter drops list of codes from a pretrained Sentence Entity Resolver Model.

For more examples, please check [Fine-Tuning Sentence Entity Resolver Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/13.1.Finetuning_Sentence_Entity_Resolver_Model.ipynb)

#### Updated ICD10CM Entity Resolver Models

We have updated `sbiobertresolve_icd10cm_augmented` model with [ICD10CM 2022 Dataset](https://www.cdc.gov/nchs/icd/icd10cm.htm) and `sbiobertresolve_icd10cm_augmented_billable_hcc` model by dropping invalid codes.

#### Updated NER Profiling Pretrained Pipelines

We have updated `ner_profiling_clinical` and `ner_profiling_biobert` pretrained pipelines by adding new clinical NER models and NER model outputs to the previous versions. In this way, you can see all the NER labels of tokens. For examples, please check [NER Profiling Pretrained Pipeline Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/11.2.Pretrained_NER_Profiling_Pipelines.ipynb).

#### New ChunkSentenceSplitter Annotator

+ We are releasing `ChunkSentenceSplitter` annotator that splits documents or sentences by chunks provided. Splitted parts can be named with the splitting chunks. By using this annotator, you can do some some tasks like splitting clinical documents according into sections in accordance with CDA (Clinical Document Architecture).

*Example* :

```python
...
ner_converter = NerConverter() \
      .setInputCols(["document", "token", "ner"]) \
      .setOutputCol("ner_chunk")\
      .setWhiteList(["Header"])

chunkSentenceSplitter = ChunkSentenceSplitter()\
    .setInputCols("ner_chunk","document")\
    .setOutputCol("paragraphs")\
    .setGroupBySentences(True) \
    .setDefaultEntity("Intro") \
    .setInsertChunk(False)        
...

text = ["""INTRODUCTION: Right pleural effusion and suspected malignant mesothelioma.
PREOPERATIVE DIAGNOSIS:  Right pleural effusion and suspected malignant mesothelioma.
POSTOPERATIVE DIAGNOSIS: Right pleural effusion, suspected malignant mesothelioma.
PROCEDURE:  Right VATS pleurodesis and pleural biopsy."""]

results = pipeline_model.transform(df)
```

*Results* :

```bash
+----------------------------------------------------------------------+------+
|                                                                result|entity|
+----------------------------------------------------------------------+------+
|INTRODUCTION: Right pleural effusion and suspected malignant mesoth...|Header|
|PREOPERATIVE DIAGNOSIS:  Right pleural effusion and suspected malig...|Header|
|POSTOPERATIVE DIAGNOSIS: Right pleural effusion, suspected malignan...|Header|
|                 PROCEDURE:  Right VATS pleurodesis and pleural biopsy|Header|
+----------------------------------------------------------------------+------+
```

- By using `.setInsertChunk()` parameter you can remove the chunk from splitted parts.

*Example* :

```python
chunkSentenceSplitter = ChunkSentenceSplitter()\
    .setInputCols("ner_chunk","document")\
    .setOutputCol("paragraphs")\
    .setGroupBySentences(True) \
    .setDefaultEntity("Intro") \
    .setInsertChunk(False)

paragraphs = chunkSentenceSplitter.transform(results)

df = paragraphs.selectExpr("explode(paragraphs) as result")\
               .selectExpr("result.result",
                           "result.metadata.entity",
                           "result.metadata.splitter_chunk")

```

*Results* :

```bash
+--------------------------------------------------+------+------------------------+
|                                            result|entity|          splitter_chunk|
+--------------------------------------------------+------+------------------------+
| Right pleural effusion and suspected malignant...|Header|           INTRODUCTION:|
|  Right pleural effusion and suspected malignan...|Header| PREOPERATIVE DIAGNOSIS:|
| Right pleural effusion, suspected malignant me...|Header|POSTOPERATIVE DIAGNOSIS:|
|         Right VATS pleurodesis and pleural biopsy|Header|              PROCEDURE:|
+--------------------------------------------------+------+------------------------+
```


#### Updated Spark NLP For Healthcare Notebooks

- [NER Profiling Pretrained Pipeline Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/11.2.Pretrained_NER_Profiling_Pipelines.ipynb) .
- [Fine-Tuning Sentence Entity Resolver Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/13.1.Finetuning_Sentence_Entity_Resolver_Model.ipynb)


**To see more, please check : [Spark NLP Healthcare Workshop Repo](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings/Healthcare)**

<div class="prev_ver h3-box" markdown="1">

## Versions

</div>

<ul class="pagination">
    <li>
        <a href="release_notes_3_3_1">Version 3.3.1</a>
    </li>
    <li>
        <strong>Version 3.3.2</strong>
    </li>
    <li>
        <a href="release_notes_3_3_4">Version 3.3.4</a>
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
    <li class="active"><a href="release_notes_3_3_2">3.3.2</a></li>
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