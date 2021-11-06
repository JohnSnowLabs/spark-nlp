---
layout: docs
header: true
seotitle: Spark NLP for Healthcare | John Snow Labs
title: Spark NLP for Healthcare Release Notes
permalink: /docs/en/licensed_release_notes
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

test_sentence = 'Mujer de 28 años con antecedentes de diabetes mellitus gestacional diagnosticada ocho años antes de la presentación y posterior diabetes mellitus tipo dos (DM2), un episodio previo de pancreatitis inducida por HTG tres años antes de la presentación, asociado con una hepatitis aguda, y obesidad con un índice de masa corporal (IMC) de 33,5 kg / m2, que se presentó con antecedentes de una semana de poliuria, polidipsia, falta de apetito y vómitos. Dos semanas antes de la presentación, fue tratada con un ciclo de cinco días de amoxicilina por una infección del tracto respiratorio. Estaba tomando metformina, glipizida y dapagliflozina para la DM2 y atorvastatina y gemfibrozil para la HTG. Había estado tomando dapagliflozina durante seis meses en el momento de la presentación. El examen físico al momento de la presentación fue significativo para la mucosa oral seca; significativamente, su examen abdominal fue benigno sin dolor a la palpación, protección o rigidez. Los hallazgos de laboratorio pertinentes al ingreso fueron: glucosa sérica 111 mg / dl, bicarbonato 18 mmol / l, anión gap 20, creatinina 0,4 mg / dl, triglicéridos 508 mg / dl, colesterol total 122 mg / dl, hemoglobina glucosilada (HbA1c) 10%. y pH venoso 7,27. La lipasa sérica fue normal a 43 U / L. Los niveles séricos de acetona no pudieron evaluarse ya que las muestras de sangre se mantuvieron hemolizadas debido a una lipemia significativa. La paciente ingresó inicialmente por cetosis por inanición, ya que refirió una ingesta oral deficiente durante los tres días previos a la admisión. Sin embargo, la química sérica obtenida seis horas después de la presentación reveló que su glucosa era de 186 mg / dL, la brecha aniónica todavía estaba elevada a 21, el bicarbonato sérico era de 16 mmol / L, el nivel de triglicéridos alcanzó un máximo de 2050 mg / dL y la lipasa fue de 52 U / L. Se obtuvo el nivel de β-hidroxibutirato y se encontró que estaba elevado a 5,29 mmol / L; la muestra original se centrifugó y la capa de quilomicrones se eliminó antes del análisis debido a la interferencia de la turbidez causada por la lipemia nuevamente. El paciente fue tratado con un goteo de insulina para euDKA y HTG con una reducción de la brecha aniónica a 13 y triglicéridos a 1400 mg / dL, dentro de las 24 horas. Se pensó que su euDKA fue precipitada por su infección del tracto respiratorio en el contexto del uso del inhibidor de SGLT2. La paciente fue atendida por el servicio de endocrinología y fue dada de alta con 40 unidades de insulina glargina por la noche, 12 unidades de insulina lispro con las comidas y metformina 1000 mg dos veces al día. Se determinó que todos los inhibidores de SGLT2 deben suspenderse indefinidamente. Tuvo un seguimiento estrecho con endocrinología post alta.'
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
    
er = SentenceEntityResolverModel.pretrained("roberta_base_biomedical_snomed", "es", "clinical/models")\
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

test_sentence = 'Mujer de 28 años con antecedentes de diabetes mellitus gestacional diagnosticada ocho años antes de la presentación y posterior diabetes mellitus tipo dos (DM2), un episodio previo de pancreatitis inducida por HTG tres años antes de la presentación, asociado con una hepatitis aguda, y obesidad con un índice de masa corporal (IMC) de 33,5 kg / m2, que se presentó con antecedentes de una semana de poliuria, polidipsia, falta de apetito y vómitos. Dos semanas antes de la presentación, fue tratada con un ciclo de cinco días de amoxicilina por una infección del tracto respiratorio. Estaba tomando metformina, glipizida y dapagliflozina para la DM2 y atorvastatina y gemfibrozil para la HTG. Había estado tomando dapagliflozina durante seis meses en el momento de la presentación. El examen físico al momento de la presentación fue significativo para la mucosa oral seca; significativamente, su examen abdominal fue benigno sin dolor a la palpación, protección o rigidez. Los hallazgos de laboratorio pertinentes al ingreso fueron: glucosa sérica 111 mg / dl, bicarbonato 18 mmol / l, anión gap 20, creatinina 0,4 mg / dl, triglicéridos 508 mg / dl, colesterol total 122 mg / dl, hemoglobina glucosilada (HbA1c) 10%. y pH venoso 7,27. La lipasa sérica fue normal a 43 U / L. Los niveles séricos de acetona no pudieron evaluarse ya que las muestras de sangre se mantuvieron hemolizadas debido a una lipemia significativa. La paciente ingresó inicialmente por cetosis por inanición, ya que refirió una ingesta oral deficiente durante los tres días previos a la admisión. Sin embargo, la química sérica obtenida seis horas después de la presentación reveló que su glucosa era de 186 mg / dL, la brecha aniónica todavía estaba elevada a 21, el bicarbonato sérico era de 16 mmol / L, el nivel de triglicéridos alcanzó un máximo de 2050 mg / dL y la lipasa fue de 52 U / L. Se obtuvo el nivel de β-hidroxibutirato y se encontró que estaba elevado a 5,29 mmol / L; la muestra original se centrifugó y la capa de quilomicrones se eliminó antes del análisis debido a la interferencia de la turbidez causada por la lipemia nuevamente. El paciente fue tratado con un goteo de insulina para euDKA y HTG con una reducción de la brecha aniónica a 13 y triglicéridos a 1400 mg / dL, dentro de las 24 horas. Se pensó que su euDKA fue precipitada por su infección del tracto respiratorio en el contexto del uso del inhibidor de SGLT2. La paciente fue atendida por el servicio de endocrinología y fue dada de alta con 40 unidades de insulina glargina por la noche, 12 unidades de insulina lispro con las comidas y metformina 1000 mg dos veces al día. Se determinó que todos los inhibidores de SGLT2 deben suspenderse indefinidamente. Tuvo un seguimiento estrecho con endocrinología post alta.'

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


## 3.3.1
We are glad to announce that Spark NLP Healthcare 3.3.1 has been released!.

#### Highlights
+ New ChunkKeyPhraseExtraction Annotator
+ New BERT-Based NER Models
+ New UMLS Sentence Entity Resolver Models
+ Updated RxNorm Entity Resolver Model (Dropping Invalid Codes)
+ New showVersion() Method in Compatibility Class
+ New Docker Images for Spark NLP for Healthcare and Spark OCR
+ New and Updated Deidentification() Parameters
+ New Python API Documentation
+ Updated Spark NLP For Healthcare Notebooks and New Notebooks 

#### New ChunkKeyPhraseExtraction Annotator

We are releasing `ChunkKeyPhraseExtraction` annotator that leverages Sentence BERT embeddings to select keywords and key phrases that are most similar to a document. This annotator can be fed by either the output of NER model, NGramGenerator or YAKE, and could be used to generate similarity scores for each NER chunk that is coming out of any (clinical) NER model. That is, you can now sort your clinical entities by the importance of them with respect to document or sentence that they live in. Additionally, you can also use this new annotator to grab new clinical chunks that are missed by a pretrained NER model as well as summarizing the whole document into a few important sentences or phrases.

You can find more examples in [ChunkKeyPhraseExtraction notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/9.Chunk_Key_Phrase_Extraction.ipynb)

*Example* :

```bash
...
ngram_ner_key_phrase_extractor = ChunkKeyPhraseExtraction.pretrained("sbert_jsl_medium_uncased ", "en", "clinical/models")\
    .setTopN(5) \
    .setDivergence(0.4)\
    .setInputCols(["sentences", "merged_chunks"])\
    .setOutputCol("key_phrases")
...

text = "A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus ( T2DM ), one prior episode of HTG-induced pancreatitis three years prior to presentation , associated with an acute hepatitis , and obesity with a body mass index ( BMI ) of 33.5 kg/m2 , presented with a one-week history of polyuria , polydipsia , poor appetite , and vomiting . Two weeks prior to presentation, she was treated with a five-day course of amoxicillin for a respiratory tract infection. She was on metformin , glipizide , and dapagliflozin for T2DM and atorvastatin and gemfibrozil for HTG. She had been on dapagliflozin for six months at the time of presentation . Physical examination on presentation was significant for dry oral mucosa ; significantly, her abdominal examination was benign with no tenderness , guarding , or rigidity . Pertinent laboratory findings on admission were: serum glucose 111 mg/dl , bicarbonate 18 mmol/l , anion gap 20 , creatinine 0.4 mg/dL , triglycerides 508 mg/dL , total cholesterol 122 mg/dL , glycated hemoglobin ( HbA1c ) 10% , and venous pH 7.27. Serum lipase was normal at 43 U/L . Serum acetone levels could not be assessed as blood samples kept hemolyzing due to significant lipemia ."


textDF = spark.createDataFrame([[text]]).toDF("text")
ngram_ner_results =  ngram_ner_pipeline.transform(textDF)
```

*Results* :

```bash
+--------------------------+------+-------------------+-------------------+--------+
|key_phrase                |source|DocumentSimilarity |MMRScore           |sentence|
+--------------------------+------+-------------------+-------------------+--------+
|type two diabetes mellitus|NER   |0.7639750686118073 |0.4583850593816694 |0       |
|HTG-induced pancreatitis  |ngrams|0.66933222897749   |0.10416352343367463|0       |
|vomiting                  |ngrams|0.5824238088130589 |0.14864183399720493|0       |
|history polyuria          |ngrams|0.46337313737310987|0.0959500325843913 |0       |
|28-year-old female        |ngrams|0.31692529374916967|0.10043002919664669|0       |
+--------------------------+------+-------------------+-------------------+--------+
```

#### New BERT-Based NER Models

We have two new BERT-Based token classifier NER models.

+ `bert_token_classifier_ner_chemicals` : This model is BERT-based version of `ner_chemicals` model and can detect chemical compounds (`CHEM`) in the medical texts.

*Metrics* :

```bash
              precision    recall  f1-score   support
      B-CHEM       0.94      0.92      0.93     30731
      I-CHEM       0.95      0.93      0.94     31270
    accuracy                           0.99     62001
   macro avg       0.96      0.95      0.96     62001
weighted avg       0.99      0.93      0.96     62001
```


*Example* :

```python 
...
tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_ner_chemicals", "en", "clinical/models")\
    .setInputCols("token", "document")\
    .setOutputCol("ner")\
    .setCaseSensitive(True) 
...

test_sentence = """The results have shown that the product p - choloroaniline is not a significant factor in chlorhexidine - digluconate associated erosive cystitis. A high percentage of kanamycin - colistin and povidone - iodine irrigations were associated with erosive cystitis."""
result = p_model.transform(spark.createDataFrame([[test_sentence]]).toDF("text"))
```

*Results* :

```bash
+---------------------------+---------+
|chunk                      |ner_label|
+---------------------------+---------+
|p - choloroaniline         |CHEM     |
|chlorhexidine - digluconate|CHEM     |
|kanamycin                  |CHEM     |
|colistin                   |CHEM     |
|povidone - iodine          |CHEM     |
+---------------------------+---------+
```

+ `bert_token_classifier_ner_chemprot` : This model is BERT-based version of `ner_chemprot_clinical` model and can detect chemical compounds and genes (`CHEMICAL`, `GENE-Y`, `GENE-N`) in the medical texts.

*Metrics* :

```bash
              precision    recall  f1-score   support
  B-CHEMICAL       0.80      0.79      0.80      8649
    B-GENE-N       0.53      0.56      0.54      2752
    B-GENE-Y       0.71      0.73      0.72      5490
  I-CHEMICAL       0.82      0.79      0.81      1313
    I-GENE-N       0.62      0.62      0.62      1993
    I-GENE-Y       0.75      0.72      0.74      2420
    accuracy                           0.96     22617
   macro avg       0.75      0.74      0.75     22617
weighted avg       0.83      0.73      0.78     22617
```

*Example* :

```bash
...
tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_ner_chemprot", "en", "clinical/models")\
    .setInputCols("token", "document")\
    .setOutputCol("ner")\
    .setCaseSensitive(True)
...

test_sentence = "Keratinocyte growth factor and acidic fibroblast growth factor are mitogens for primary cultures of mammary epithelium."
result = p_model.transform(spark.createDataFrame([[test_sentence]]).toDF("text"))
```

*Results* :

```bash
+-------------------------------+---------+
|chunk                          |ner_label|
+-------------------------------+---------+
|Keratinocyte growth factor     |GENE-Y   |
|acidic fibroblast growth factor|GENE-Y   |
+-------------------------------+---------+
```


#### New UMLS Sentence Entity Resolver Models

We are releasing two new UMLS Sentence Entity Resolver models trained on 2021AB UMLS dataset and map clinical entities to UMLS CUI codes.

+ `sbiobertresolve_umls_disease_syndrome` : This model is trained on the `Disease` or `Syndrome` category using `sbiobert_base_cased_mli` embeddings.

*Example* :

```bash
...
resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_umls_disease_syndrome","en", "clinical/models") \
     .setInputCols(["ner_chunk", "sbert_embeddings"]) \
     .setOutputCol("resolution")\
     .setDistanceFunction("EUCLIDEAN")
...

data = spark.createDataFrame([["""A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus (T2DM), one prior episode of HTG-induced pancreatitis three years prior to presentation, associated with an acute hepatitis, and obesity with a body mass index (BMI) of 33.5 kg/m2, presented with a one-week history of polyuria, polydipsia, poor appetite, and vomiting."""]]).toDF("text")
results = model.fit(data).transform(data)

```

*Results* :

```bash
|    | chunk                                 | code     | code_description                      | all_k_codes                                                  | all_k_codes_desc                                                                                                                                                                                         |
|---:|:--------------------------------------|:---------|:--------------------------------------|:-------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|  0 | gestational diabetes mellitus         | C0085207 | gestational diabetes mellitus         | ['C0085207', 'C0032969', 'C2063017', 'C1283034', 'C0271663'] | ['gestational diabetes mellitus', 'pregnancy diabetes mellitus', 'pregnancy complicated by diabetes mellitus', 'maternal diabetes mellitus', 'gestational diabetes mellitus, a2']                        |
|  1 | subsequent type two diabetes mellitus | C0348921 | pre-existing type 2 diabetes mellitus | ['C0348921', 'C1719939', 'C0011860', 'C0877302', 'C0271640'] | ['pre-existing type 2 diabetes mellitus', 'disorder associated with type 2 diabetes mellitus', 'diabetes mellitus, type 2', 'insulin-requiring type 2 diabetes mellitus', 'secondary diabetes mellitus'] |
|  2 | HTG-induced pancreatitis              | C0376670 | alcohol-induced pancreatitis          | ['C0376670', 'C1868971', 'C4302243', 'C0267940', 'C2350449'] | ['alcohol-induced pancreatitis', 'toxic pancreatitis', 'igg4-related pancreatitis', 'hemorrhage pancreatitis', 'graft pancreatitis']                                                                     |
|  3 | an acute hepatitis                    | C0019159 | acute hepatitis                       | ['C0019159', 'C0276434', 'C0267797', 'C1386146', 'C2063407'] | ['acute hepatitis a', 'acute hepatitis a', 'acute hepatitis', 'acute infectious hepatitis', 'acute hepatitis e']                                                                                         |
|  4 | obesity                               | C0028754 | obesity                               | ['C0028754', 'C0342940', 'C0342942', 'C0857116', 'C1561826'] | ['obesity', 'abdominal obesity', 'generalized obesity', 'obesity gross', 'overweight and obesity']                                                                                                       |
|  5 | polyuria                              | C0018965 | hematuria                             | ['C0018965', 'C0151582', 'C3888890', 'C0268556', 'C2936921'] | ['hematuria', 'uricosuria', 'polyuria-polydipsia syndrome', 'saccharopinuria', 'saccharopinuria']                                                                                                        |
|  6 | polydipsia                            | C0268813 | primary polydipsia                    | ['C0268813', 'C0030508', 'C3888890', 'C0393777', 'C0206085'] | ['primary polydipsia', 'parasomnia', 'polyuria-polydipsia syndrome', 'hypnogenic paroxysmal dystonias', 'periodic hypersomnias']                                                                         |
|  7 | poor appetite                         | C0003123 | lack of appetite                      | ['C0003123', 'C0011168', 'C0162429', 'C1282895', 'C0039338'] | ['lack of appetite', 'poor swallowing', 'poor nutrition', 'neurologic unpleasant taste', 'taste dis']                                                                                                    |
|  8 | vomiting                              | C0152164 | periodic vomiting                     | ['C0152164', 'C0267172', 'C0152517', 'C0011119', 'C0152227'] | ['periodic vomiting', 'habit vomiting', 'viral vomiting', 'choking', 'tearing']                                                                                                                          |
```

+ `sbiobertresolve_umls_clinical_drugs` : This model is trained on the `Clinical Drug` category using `sbiobert_base_cased_mli` embeddings.

*Example* :

```bash
...
resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_umls_clinical_drugs","en", "clinical/models") \
     .setInputCols(["ner_chunk", "sbert_embeddings"]) \
     .setOutputCol("resolution")\
     .setDistanceFunction("EUCLIDEAN")
...

data = spark.createDataFrame([["""She was immediately given hydrogen peroxide 30 mg to treat the infection on her leg, and has been advised Neosporin Cream for 5 days. She has a history of taking magnesium hydroxide 100mg/1ml and metformin 1000 mg."""]]).toDF("text")
results = model.fit(data).transform(data)

```

*Results* :

```bash
|    | chunk                         | code     | code_description           | all_k_codes                                                  | all_k_codes_desc                                                                                                                                                                        |
|---:|:------------------------------|:---------|:---------------------------|:-------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|  0 | hydrogen peroxide 30 mg       | C1126248 | hydrogen peroxide 30 mg/ml | ['C1126248', 'C0304655', 'C1605252', 'C0304656', 'C1154260'] | ['hydrogen peroxide 30 mg/ml', 'hydrogen peroxide solution 30%', 'hydrogen peroxide 30 mg/ml [proxacol]', 'hydrogen peroxide 30 mg/ml cutaneous solution', 'benzoyl peroxide 30 mg/ml'] |
|  1 | Neosporin Cream               | C0132149 | neosporin cream            | ['C0132149', 'C0358174', 'C0357999', 'C0307085', 'C0698810'] | ['neosporin cream', 'nystan cream', 'nystadermal cream', 'nupercainal cream', 'nystaform cream']                                                                                        |
|  2 | magnesium hydroxide 100mg/1ml | C1134402 | magnesium hydroxide 100 mg | ['C1134402', 'C1126785', 'C4317023', 'C4051486', 'C4047137'] | ['magnesium hydroxide 100 mg', 'magnesium hydroxide 100 mg/ml', 'magnesium sulphate 100mg/ml injection', 'magnesium sulfate 100 mg', 'magnesium sulfate 100 mg/ml']                     |
|  3 | metformin 1000 mg             | C0987664 | metformin 1000 mg          | ['C0987664', 'C2719784', 'C0978482', 'C2719786', 'C4282269'] | ['metformin 1000 mg', 'metformin hydrochloride 1000 mg', 'metformin hcl 1000mg tab', 'metformin hydrochloride 1000 mg [fortamet]', 'metformin hcl 1000mg sa tab']                       |

```

#### Updated RxNorm Entity Resolver Model (Dropping Invalid Codes)

`sbiobertresolve_rxnorm` model was updated by dropping invalid codes using 02 August 2021 RxNorm dataset.

#### New showVersion() Method in Compatibility Class

We added the `.showVersion()` method in our Compatibility class that shows the name of the models and the version in a pretty way.

```python
compatibility = Compatibility()
compatibility.showVersion('sentence_detector_dl_healthcare')
```
After the execution you will see the following table,

```bash
+---------------------------------+------+---------+
| Pipeline/Model                  | lang | version |
+---------------------------------+------+---------+
| sentence_detector_dl_healthcare |  en  | 2.6.0   |
| sentence_detector_dl_healthcare |  en  | 2.7.0   |
| sentence_detector_dl_healthcare |  en  | 3.2.0   |
+---------------------------------+------+---------+
```

#### New Docker Images for Spark NLP for Healthcare and Spark OCR

We are releasing new Docker Images for Spark NLP for Healthcare and Spark OCR containing a jupyter environment. Users having a valid license can run the image on their local system, and connect to pre-configured jupyter instance without installing the library on their local system.

**Spark NLP for Healthcare Docker Image**

For running Spark NLP for Healthcare inside a container:

- Instructions: [Spark NLP for Healthcare Docker Image](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/jupyter/docker_image_nlp_hc)

- Video Instructions: [Youtube Video](https://www.youtube.com/watch?v=tgN0GZGMVJk)

**Spark NLP for Healthcare & OCR Docker Image**

For users who want to run Spark OCR and then feed the output of OCR pipeline to healthcare modules to process further:

- Instructions: [Spark NLP for Healthcare & OCR Docker Image](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/jupyter/docker_image_ocr)

#### New and Updated Deidentification() Parameters

*New Parameter* :
+ `setBlackList()` : List of entities **ignored** for masking or obfuscation.The default values are: `SSN`, `PASSPORT`, `DLN`, `NPI`, `C_CARD`, `IBAN`, `DEA`.

*Updated Parameter* :

+ `.setObfuscateRefSource()` : It was set `faker` as default.

#### New Python API Documentation

We have new Spark NLP for Healthcare [Python API Documentation](https://nlp.johnsnowlabs.com/licensed/api/python/) . This page contains information how to use the library with Python examples.

#### Updated Spark NLP For Healthcare Notebooks and New Notebooks

- New [BertForTokenClassification NER Model Training with Transformers Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.6.BertForTokenClassification_NER_SparkNLP_with_Transformers.ipynb) for showing how to train a BertForTokenClassification NER model with transformers and then import into Spark NLP.

- New [ChunkKeyPhraseExtraction notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/9.Chunk_Key_Phrase_Extraction.ipynb) for showing how to get chunk key phrases using `ChunkKeyPhraseExtraction`.

- Updated all [Spark NLP For Healthcare Notebooks](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings/Healthcare) with v3.3.0 by adding the new features.



**To see more, please check : [Spark NLP Healthcare Workshop Repo](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings/Healthcare)**


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


## 3.2.3	
We are glad to announce that Spark NLP Healthcare 3.2.3 has been released!.

#### Highlights
+ New BERT-Based Deidentification NER Model
+ New Sentence Entity Resolver Models For German Language
+ New Spell Checker Model For Drugs
+ Allow To Use Disambiguator Pretrained Model
+ Allow To Use Seeds in StructuredDeidentification
+ Added Compatibility with Tensorflow 1.15 For Graph Generation.
+ New Setup Videos 

#### New BERT-Based Deidentification NER Model

We have a new `bert_token_classifier_ner_deid` model that is BERT-based version of `ner_deid_subentity_augmented` and annotates text to find protected health information that may need to be de-identified. It can detect 23 different entities (`MEDICALRECORD`, `ORGANIZATION`, `DOCTOR`, `USERNAME`, `PROFESSION`, `HEALTHPLAN`, `URL`, `CITY`, `DATE`, `LOCATION-OTHER`, `STATE`, `PATIENT`, `DEVICE`, `COUNTRY`, `ZIP`, `PHONE`, `HOSPITAL`, `EMAIL`, `IDNUM`, `SREET`, `BIOID`, `FAX`, `AGE`).

*Example*:

```bash
documentAssembler = DocumentAssembler()\
  .setInputCol("text")\
  .setOutputCol("document")
  
tokenizer = Tokenizer()\
  .setInputCols("document")\
  .setOutputCol("token")
  
tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_ner_deid", "en")\
  .setInputCols("token", "document")\
  .setOutputCol("ner")\
  .setCaseSensitive(True)
  
ner_converter = NerConverter()\
  .setInputCols(["document","token","ner"])\
  .setOutputCol("ner_chunk")
        
pipeline =  Pipeline(stages=[documentAssembler, tokenizer, tokenClassifier, ner_converter])
p_model = pipeline.fit(spark.createDataFrame(pd.DataFrame({'text': ['']})))

text = """A. Record date : 2093-01-13, David Hale, M.D. Name : Hendrickson, Ora MR. # 7194334. PCP : Oliveira, non-smoking. Cocke County Baptist Hospital. 0295 Keats Street. Phone +1 (302) 786-5227. Patient's complaints first surfaced when he started working for Brothers Coal-Mine."""
result = p_model.transform(spark.createDataFrame(pd.DataFrame({'text': [text]})))
```

*Results*:

```bash
+-----------------------------+-------------+
|chunk                        |ner_label    |
+-----------------------------+-------------+
|2093-01-13                   |DATE         |
|David Hale                   |DOCTOR       |
|Hendrickson, Ora             |PATIENT      |
|7194334                      |MEDICALRECORD|
|Oliveira                     |PATIENT      |
|Cocke County Baptist Hospital|HOSPITAL     |
|0295 Keats Street            |STREET       |
|302) 786-5227                |PHONE        |
|Brothers Coal-Mine           |ORGANIZATION |
+-----------------------------+-------------+
```

#### New Sentence Entity Resolver Models For German Language

We are releasing two new Sentence Entity Resolver Models for German language that use `sent_bert_base_cased` (de) embeddings.

+ `sbertresolve_icd10gm` : This model maps extracted medical entities to ICD10-GM codes for the German language.

*Example*:

```bash
documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("ner_chunk")
    
sbert_embedder = BertSentenceEmbeddings.pretrained("sent_bert_base_cased", "de")\
    .setInputCols(["ner_chunk"])\
    .setOutputCol("sbert_embeddings")

icd10gm_resolver = SentenceEntityResolverModel.pretrained("sbertresolve_icd10gm", "de", "clinical/models")\
    .setInputCols(["ner_chunk", "sbert_embeddings"])\
    .setOutputCol("icd10gm_code")

icd10gm_pipelineModel = PipelineModel( stages = [documentAssembler, sbert_embedder, icd10gm_resolver])

icd_lp = LightPipeline(icd10gm_pipelineModel)
icd_lp.fullAnnotate("Dyspnoe")
```

*Results* :

|chunk|code|resolutions|all_codes|all_distances|
|-|-|-|-|-|
| Dyspnoe | C671 | Dyspnoe, Schlafapnoe, Dysphonie, Frühsyphilis, Hyperzementose, Hypertrichose, ...  | [R06.0, G47.3, R49.0, A51, K03.4, L68, ...] | [0.0000, 2.5602, 3.0529, 3.3310, 3.4645, 3.7148, ...] |

+ `sbertresolve_snomed` : This model maps extracted medical entities to SNOMED codes for the German language.

*Example*:

```bash
documentAssembler = DocumentAssembler()\ 
    .setInputCol("text")\ 
    .setOutputCol("ner_chunk")
    
sbert_embedder = BertSentenceEmbeddings.pretrained("sent_bert_base_cased", "de")\
    .setInputCols(["ner_chunk"])\
    .setOutputCol("sbert_embeddings")

snomed_resolver = SentenceEntityResolverModel.pretrained("sbertresolve_snomed", "de", "clinical/models")\
    .setInputCols(["ner_chunk", "sbert_embeddings"])\
    .setOutputCol("snomed_code")

snomed_pipelineModel = PipelineModel( stages = [ documentAssembler, sbert_embedder, snomed_resolver])

snomed_lp = LightPipeline(snomed_pipelineModel)
snomed_lp.fullAnnotate("Bronchialkarzinom ")
```

*Results* :

|chunk|code|resolutions|all_codes|all_distances|
|-|-|-|-|-|
| Bronchialkarzinom  | 22628 | Bronchialkarzinom, Bronchuskarzinom, Rektumkarzinom, Klavikulakarzinom, Lippenkarzinom, Urothelkarzinom, ...  | [22628, 111139, 18116, 107569, 18830, 22909, ...] | [0.0000, 0.0073, 0.0090, 0.0098, 0.0098, 0.0102, ...] |

#### New Spell Checker Model For Drugs

We are releasing new `spellcheck_drug_norvig` model that detects and corrects spelling errors of drugs in a text based on the Norvig's approach.

*Example* :

```bash
documentAssembler = DocumentAssembler()\ 
    .setInputCol("text")\ 
    .setOutputCol("document")
    
tokenizer = Tokenizer()
    .setInputCols("document")\
    .setOutputCol("token")

spell = NorvigSweetingModel.pretrained("spellcheck_drug_norvig", "en", "clinical/models")\
    .setInputCols("token")
    .setOutputCol("spell")\

pipeline = Pipeline( stages = [documentAssembler,
tokenizer, spell])

model = pipeline.fit(spark.createDataFrame([['']]).toDF('text')) 
lp = LightPipeline(model)

lp.annotate("You have to take Neutrcare and colfosrinum and a bit of Fluorometholne & Ribotril")
```

*Results* :

```bash
Original text  : You have to take Neutrcare and colfosrinum and a bit of fluorometholne & Ribotril
Corrected text : You have to take Neutracare and colforsinum and a bit of fluorometholone & Rivotril

```

#### Allow to use Disambiguator pretrained model.

Now we can use the NerDisambiguatorModel as a pretrained model to disambiguate person entities.

```python
 text = "The show also had a contestant named Brad Pitt" \
        + "who later defeated Christina Aguilera on the way to become Female Vocalist Champion in the 1989 edition of Star Search in the United States. "
 data = SparkContextForTest.spark.createDataFrame([
     [text]]) \
     .toDF("text").cache()
 da = DocumentAssembler().setInputCol("text").setOutputCol("document")
 
 sd = SentenceDetector().setInputCols("document").setOutputCol("sentence")
 
 tk = Tokenizer().setInputCols("sentence").setOutputCol("token")
 
 emb = WordEmbeddingsModel.pretrained().setOutputCol("embs")
 
 semb = SentenceEmbeddings().setInputCols("sentence", "embs").setOutputCol("sentence_embeddings")
 
 ner = NerDLModel.pretrained().setInputCols("sentence", "token", "embs").setOutputCol("ner")
 
 nc = NerConverter().setInputCols("sentence", "token", "ner").setOutputCol("ner_chunk").setWhiteList(["PER"])
 
 NerDisambiguatorModel.pretrained().setInputCols("ner_chunk", "sentence_embeddings").setOutputCol("disambiguation")
 
 pl = Pipeline().setStages([da, sd, tk, emb, semb, ner, nc, disambiguator])
 
 data = pl.fit(data).transform(data)
 data.select("disambiguation").show(10, False)

```

```bash
+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|disambiguation                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|[[disambiguation, 65, 82, http://en.wikipedia.org/?curid=144171, http://en.wikipedia.org/?curid=6636454, [chunk -> Christina Aguilera, titles -> christina aguilera ::::: christina aguilar, links -> http://en.wikipedia.org/?curid=144171 ::::: http://en.wikipedia.org/?curid=6636454, beginInText -> 65, scores -> 0.9764155197864447, 0.9727793647472524, categories -> Musicians, Singers, Actors, Businesspeople, Musicians, Singers, ids -> 144171, 6636454, endInText -> 82], []]]|
+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

----------------------

```
#### Allow to use seeds in StructuredDeidentification

Now, we can use a seed for a specific column. The seed is used to randomly select the entities used during obfuscation mode. By providing the same seed, you can replicate the same mapping multiple times.

```python
df = spark.createDataFrame([
            ["12", "12", "Juan García"],
            ["24", "56", "Will Smith"],
            ["56", "32", "Pedro Ximénez"]
        ]).toDF("ID1", "ID2", "NAME")
        
obfuscator = StructuredDeidentification(spark=spark, columns={"ID1": "ID", "ID2": "ID", "NAME": "PATIENT"},
                                                columnsSeed={"ID1": 23, "ID2": 23},
                                                obfuscateRefSource="faker")
result = obfuscator.obfuscateColumns(df)
result.show(truncate=False)      
```

```bash
+----------+----------+----------------+
|ID1       |ID2       |NAME            |
+----------+----------+----------------+
|[D3379888]|[D3379888]|[Raina Cleaves] |
|[R8448971]|[M8851891]|[Jennell Barre] |
|[M8851891]|[L5448098]|[Norene Salines]|
+----------+----------+----------------+

Here, you can see that as we have provided the same seed `23` for columns `ID1`, and `ID2`, the number `12` which is appears twice in the first row is mapped to the same randomly generated id `D3379888` each time.
```
#### Added compatibility with Tensorflow 1.15 for graph generation
Some users reported problems while using graphs generated by Tensorflow 2.x. We provide compatibility with Tensorflow 1.15 in the `tf_graph_1x` module, that can be used like this,

```
from sparknlp_jsl.training import tf_graph_1x

```

In next releases, we will provide full support for graph generation using Tensorflow 2.x.

#### New Setup Videos

Now we have videos showing how to setup Spark NLP, Spark NLP for Healthcare and Spark OCR on UBUNTU.

+ [How to Setup Spark NLP on UBUNTU](https://www.youtube.com/watch?v=ZnFENM-yNfQ)
+ [How to Setup Spark NLP for HEALTHCARE on UBUNTU](https://www.youtube.com/watch?v=yKnF-_oz0GE)
+ [How to Setup Spark OCR on UBUNTU](https://www.youtube.com/watch?v=cmt4WIcL0nI)

**To see more, please check**: [Spark NLP Healthcare Workshop Repo](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings/Healthcare)


## 3.2.2
We are glad to announce that Spark NLP Healthcare 3.2.2 has been released!.

#### Highlights
+ New NER Model For Detecting Drugs, Posology, and Administration Cycles
+ New Sentence Entity Resolver Models
+ New Router Annotator To Use Multiple Resolvers Optimally In the Same Pipeline
+ Re-Augmented Deidentification NER Model

#### New NER Model For Detecting Drugs, Posology, and Administration Cycles

We are releasing a new NER posology model `ner_posology_experimental`. This model is based on the original `ner_posology_large` model, but trained with additional clinical trials data to detect experimental drugs, experiment cycles, cycle counts, and cycles numbers. Supported Entities: `Administration`, `Cyclenumber`, `Strength`, `Cycleday`, `Duration`, `Cyclecount`, `Route`, `Form`, `Frequency`, `Cyclelength`, `Drug`, `Dosage`

*Example*:

```bash
...
word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
   .setInputCols(["sentence", "token"])\
   .setOutputCol("embeddings")
clinical_ner = MedicalNerModel.pretrained("ner_posology_experimental", "en", "clinical/models") \
   .setInputCols(["sentence", "token", "embeddings"]) \
   .setOutputCol("ner")
...
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter])
model = nlp_pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))
results = model.transform(spark.createDataFrame([["Y-90 Humanized Anti-Tac: 10 mCi (if a bone marrow transplant was part of the patient's previous therapy) or 15 mCi of yttrium labeled anti-TAC; followed by calcium trisodium Inj (Ca DTPA)..\n\nCalcium-DTPA: Ca-DTPA will be administered intravenously on Days 1-3 to clear the radioactive agent from the body."]]).toDF("text"))
```

*Results*:
```bash
|    | chunk                    |   begin |   end | entity   |
|---:|:-------------------------|--------:|------:|:---------|
|  0 | Y-90 Humanized Anti-Tac  |       0 |    22 | Drug     |
|  1 | 10 mCi                   |      25 |    30 | Dosage   |
|  2 | 15 mCi                   |     108 |   113 | Dosage   |
|  3 | yttrium labeled anti-TAC |     118 |   141 | Drug     |
|  4 | calcium trisodium Inj    |     156 |   176 | Drug     |
|  5 | Calcium-DTPA             |     191 |   202 | Drug     |
|  6 | Ca-DTPA                  |     205 |   211 | Drug     |
|  7 | intravenously            |     234 |   246 | Route    |
|  8 | Days 1-3                 |     251 |   258 | Cycleday |
```

#### New Sentence Entity Resolver Models

We have two new sentence entity resolver models trained with using `sbert_jsl_medium_uncased` embeddings.

+ `sbertresolve_rxnorm_disposition` : This model maps medication entities (like drugs/ingredients) to RxNorm codes and their dispositions using `sbert_jsl_medium_uncased` Sentence Bert Embeddings. If you look for a faster inference with just drug names (excluding dosage and strength), this version of RxNorm model would be a better alternative. In the result, look for the aux_label parameter in the metadata to get dispositions divided by `|`.

*Example*:
```bash
documentAssembler = DocumentAssembler()\
      .setInputCol("text")\
      .setOutputCol("ner_chunk")

sbert_embedder = BertSentenceEmbeddings.pretrained('sbert_jsl_medium_uncased', 'en','clinical/models')\
      .setInputCols(["ner_chunk"])\
      .setOutputCol("sbert_embeddings")
    
rxnorm_resolver = SentenceEntityResolverModel.pretrained("sbertresolve_rxnorm_disposition", "en", "clinical/models") \
      .setInputCols(["ner_chunk", "sbert_embeddings"]) \
      .setOutputCol("rxnorm_code")\
      .setDistanceFunction("EUCLIDEAN")

rxnorm_pipelineModel = PipelineModel(
    stages = [
        documentAssembler,
        sbert_embedder,
        rxnorm_resolver])

rxnorm_lp = LightPipeline(rxnorm_pipelineModel)
rxnorm_lp = LightPipeline(pipelineModel) result = rxnorm_lp.fullAnnotate("alizapride 25 mg/ml")
```
*Result*:

```bash
|    | chunks             | code   | resolutions                                                                                                                                                                            | all_codes                                                       | all_k_aux_labels                                                                                            | all_distances                                                 |
|---:|:-------------------|:-------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------|
|  0 |alizapride 25 mg/ml | 330948 | [alizapride 25 mg/ml, alizapride 50 mg, alizapride 25 mg/ml oral solution, adalimumab 50 mg/ml, adalimumab 100 mg/ml [humira], adalimumab 50 mg/ml [humira], alirocumab 150 mg/ml, ...]| [330948, 330949, 249531, 358817, 1726845, 576023, 1659153, ...] | [Dopamine receptor antagonist, Dopamine receptor antagonist, Dopamine receptor antagonist, -, -, -, -, ...] | [0.0000, 0.0936, 0.1166, 0.1525, 0.1584, 0.1567, 0.1631, ...] |
```

+ `sbertresolve_snomed_conditions` : This model maps clinical entities (domain: Conditions) to Snomed codes using `sbert_jsl_medium_uncased` Sentence Bert Embeddings.

*Example*:

```bash
documentAssembler = DocumentAssembler()\
      .setInputCol("text")\
      .setOutputCol("ner_chunk")

sbert_embedder = BertSentenceEmbeddings.pretrained('sbert_jsl_medium_uncased', 'en','clinical/models')\
      .setInputCols(["ner_chunk"])\
      .setOutputCol("sbert_embeddings")
    
snomed_resolver = SentenceEntityResolverModel.pretrained("sbertresolve_snomed_conditions", "en", "clinical/models") \
      .setInputCols(["ner_chunk", "sbert_embeddings"]) \
      .setOutputCol("snomed_code")\
      .setDistanceFunction("EUCLIDEAN")

snomed_pipelineModel = PipelineModel(
    stages = [
        documentAssembler,
        sbert_embedder,
        snomed_resolver
        ])

snomed_lp = LightPipeline(snomed_pipelineModel)
result = snomed_lp.fullAnnotate("schizophrenia")
```
*Result*:

```bash
|    | chunks        | code     | resolutions                                                                                                              | all_codes                                                            | all_distances                                        |
|---:|:--------------|:---------|:-------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------|:-----------------------------------------------------|
|  0 | schizophrenia | 58214004 | [schizophrenia, chronic schizophrenia, borderline schizophrenia, schizophrenia, catatonic, subchronic schizophrenia, ...]| [58214004, 83746006, 274952002, 191542003, 191529003, 16990005, ...] | 0.0000, 0.0774, 0.0838, 0.0927, 0.0970, 0.0970, ...] |
```

#### New Router Annotator To Use Multiple Resolvers Optimally In the Same Pipeline

Normally, when we need to use more than one sentence entity resolver models in the same pipeline, we used to hit `BertSentenceEmbeddings` annotator more than once given the number of different resolver models in the same pipeline. Now we are introducing a solution with the help of `Router` annotator that could allow us to feed all the NER chunks to `BertSentenceEmbeddings` at once and then route the output of Sentence Embeddings to different resolver models needed.

You can find an example of how to use this annotator in the updated [3.Clinical_Entity_Resolvers.ipynb Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb)

*Example*:

```bash
...
# to get PROBLEM entitis
clinical_ner = MedicalNerModel().pretrained("ner_clinical", "en", "clinical/models") \
        .setInputCols(["sentence", "token", "word_embeddings"]) \
        .setOutputCol("clinical_ner")

clinical_ner_chunk = NerConverter()\
        .setInputCols("sentence","token","clinical_ner")\
        .setOutputCol("clinical_ner_chunk")\
        .setWhiteList(["PROBLEM"])

# to get DRUG entities 
posology_ner = MedicalNerModel().pretrained("ner_posology", "en", "clinical/models") \
        .setInputCols(["sentence", "token", "word_embeddings"]) \
        .setOutputCol("posology_ner")

posology_ner_chunk = NerConverter()\
        .setInputCols("sentence","token","posology_ner")\
        .setOutputCol("posology_ner_chunk")\
        .setWhiteList(["DRUG"])

# merge the chunks into a single ner_chunk
chunk_merger = ChunkMergeApproach()\
        .setInputCols("clinical_ner_chunk","posology_ner_chunk")\
        .setOutputCol("final_ner_chunk")\
        .setMergeOverlapping(False)


# convert chunks to doc to get sentence embeddings of them
chunk2doc = Chunk2Doc().setInputCols("final_ner_chunk").setOutputCol("final_chunk_doc")


sbiobert_embeddings = BertSentenceEmbeddings.pretrained("sbiobert_base_cased_mli","en","clinical/models")\
        .setInputCols(["final_chunk_doc"])\
        .setOutputCol("sbert_embeddings")

# filter PROBLEM entity embeddings
router_sentence_icd10 = Router() \
        .setInputCols("sbert_embeddings") \
        .setFilterFieldsElements(["PROBLEM"]) \
        .setOutputCol("problem_embeddings")

# filter DRUG entity embeddings
router_sentence_rxnorm = Router() \
        .setInputCols("sbert_embeddings") \
        .setFilterFieldsElements(["DRUG"]) \
        .setOutputCol("drug_embeddings")

# use problem_embeddings only
icd_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_icd10cm_slim_billable_hcc","en", "clinical/models") \
        .setInputCols(["clinical_ner_chunk", "problem_embeddings"]) \
        .setOutputCol("icd10cm_code")\
        .setDistanceFunction("EUCLIDEAN")


# use drug_embeddings only
rxnorm_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_rxnorm","en", "clinical/models") \
        .setInputCols(["posology_ner_chunk", "drug_embeddings"]) \
        .setOutputCol("rxnorm_code")\
        .setDistanceFunction("EUCLIDEAN")


pipeline = Pipeline(stages=[
    documentAssembler,
    sentenceDetector,
    tokenizer,
    word_embeddings,
    clinical_ner,
    clinical_ner_chunk,
    posology_ner,
    posology_ner_chunk,
    chunk_merger,
    chunk2doc,
    sbiobert_embeddings,
    router_sentence_icd10,
    router_sentence_rxnorm,
    icd_resolver,
    rxnorm_resolver
])

```


#### Re-Augmented Deidentification NER Model

We re-augmented `ner_deid_subentity_augmented` deidentification NER model improving the previous metrics by 2%.

*Example*:

```bash
...
deid_ner = MedicalNerModel.pretrained("ner_deid_subentity_augmented", "en", "clinical/models") \
      .setInputCols(["sentence", "token", "embeddings"]) \
      .setOutputCol("ner")
...
nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, deid_ner, ner_converter])
model = nlpPipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

results = model.transform(spark.createDataFrame(pd.DataFrame({"text": ["""A. Record date : 2093-01-13, David Hale, M.D., Name : Hendrickson, Ora MR. # 7194334 Date : 01/13/93 PCP : Oliveira, 25 -year-old, Record date : 1-11-2000. Cocke County Baptist Hospital. 0295 Keats Street. Phone +1 (302) 786-5227."""]})))
```

*Results*:

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
|25-year-old                  |AGE          |
|1-11-2000                    |DATE         |
|Cocke County Baptist Hospital|HOSPITAL     |
|0295 Keats Street.           |STREET       |
|(302) 786-5227               |PHONE        |
|Brothers Coal-Mine           |ORGANIZATION |
+-----------------------------+-------------+
``` 


**To see more, please check:** [Spark NLP Healthcare Workshop Repo](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings/Healthcare)


## 3.2.1
We are glad to announce that Spark NLP Healthcare 3.2.1 has been released!.

#### Highlights

+ Deprecated ChunkEntityResolver.
+ New BERT-Based NER Models
+ HCC module added support for versions v22 and v23.
+ Updated Notebooks for resolvers and graph builders.
+ New TF Graph Builder.

#### New BERT-Based NER Models

We have two new BERT-based token classifier NER models. These models are the first clinical NER models that use the BertForTokenCLassification approach that was introduced in Spark NLP 3.2.0. 

+ `bert_token_classifier_ner_clinical`: This model is BERT-based version of `ner_clinical` model. This new model is 4% better than the legacy NER model (MedicalNerModel) that is based on BiLSTM-CNN-Char architecture.

*Metrics*:

```
              precision    recall  f1-score   support

     PROBLEM       0.88      0.92      0.90     30276
        TEST       0.91      0.86      0.88     17237
   TREATMENT       0.87      0.88      0.88     17298
           O       0.97      0.97      0.97    202438

    accuracy                           0.95    267249
   macro avg       0.91      0.91      0.91    267249
weighted avg       0.95      0.95      0.95    267249

```

*Example*:

```bash
documentAssembler = DocumentAssembler()\
  .setInputCol("text")\
  .setOutputCol("document")

sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models")\
       .setInputCols(["document"])\
       .setOutputCol("sentence")

tokenizer = Tokenizer()\
       .setInputCols("sentence")\
       .setOutputCol("token")

tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_ner_clinical", "en", "clinical/models")\
       .setInputCols("token", "sentence")\
       .setOutputCol("ner")\
       .setCaseSensitive(True)
  
ner_converter = NerConverter()\
        .setInputCols(["sentence","token","ner"])\
        .setOutputCol("ner_chunk")

pipeline =  Pipeline(stages=[
       documentAssembler,
       sentenceDetector,
       tokenizer,
       tokenClassifier,
       ner_converter
  ])

p_model = pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

text = 'A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus ( T2DM ), one prior episode of HTG-induced pancreatitis three years prior to presentation , associated with an acute hepatitis , and obesity with a body mass index ( BMI ) of 33.5 kg/m2 , presented with a one-week history of polyuria , polydipsia , poor appetite , and vomiting . Two weeks prior to presentation , she was treated with a five-day course of amoxicillin for a respiratory tract infection . She was on metformin , glipizide , and dapagliflozin for T2DM and atorvastatin and gemfibrozil for HTG . She had been on dapagliflozin for six months at the time of presentation . Physical examination on presentation was significant for dry oral mucosa ; significantly , her abdominal examination was benign with no tenderness , guarding , or rigidity . Pertinent laboratory findings on admission were : serum glucose 111 mg/dl , bicarbonate 18 mmol/l , anion gap 20 , creatinine 0.4 mg/dL , triglycerides 508 mg/dL , total cholesterol 122 mg/dL , glycated hemoglobin ( HbA1c ) 10% , and venous pH 7.27 . Serum lipase was normal at 43 U/L . Serum acetone levels could not be assessed as blood samples kept hemolyzing due to significant lipemia . The patient was initially admitted for starvation ketosis , as she reported poor oral intake for three days prior to admission . However , serum chemistry obtained six hours after presentation revealed her glucose was 186 mg/dL , the anion gap was still elevated at 21 , serum bicarbonate was 16 mmol/L , triglyceride level peaked at 2050 mg/dL , and lipase was 52 U/L . The β-hydroxybutyrate level was obtained and found to be elevated at 5.29 mmol/L - the original sample was centrifuged and the chylomicron layer removed prior to analysis due to interference from turbidity caused by lipemia again . The patient was treated with an insulin drip for euDKA and HTG with a reduction in the anion gap to 13 and triglycerides to 1400 mg/dL , within 24 hours . Her euDKA was thought to be precipitated by her respiratory tract infection in the setting of SGLT2 inhibitor use . The patient was seen by the endocrinology service and she was discharged on 40 units of insulin glargine at night , 12 units of insulin lispro with meals , and metformin 1000 mg two times a day . It was determined that all SGLT2 inhibitors should be discontinued indefinitely . She had close follow-up with endocrinology post discharge .'

res = p_model.transform(spark.createDataFrame([[text]]).toDF("text")).collect()

res[0]['label']
```

+ `bert_token_classifier_ner_jsl`: This model is BERT-based version of `ner_jsl` model. This new model is better than the legacy NER model (MedicalNerModel) that is based on BiLSTM-CNN-Char architecture.

*Metrics*:

```
                                    precision    recall  f1-score   support

                    Admission_Discharge       0.84      0.97      0.90       415
                                    Age       0.96      0.96      0.96      2434
                                Alcohol       0.75      0.83      0.79       145
                               Allergen       0.33      0.16      0.22        25
                                    BMI       1.00      0.77      0.87        26
                           Birth_Entity       1.00      0.17      0.29        12
                         Blood_Pressure       0.86      0.88      0.87       597
                Cerebrovascular_Disease       0.74      0.77      0.75       266
                          Clinical_Dept       0.90      0.92      0.91      2385
                   Communicable_Disease       0.70      0.59      0.64        85
                                   Date       0.95      0.98      0.96      1438
                           Death_Entity       0.83      0.83      0.83        59
                               Diabetes       0.95      0.95      0.95       350
                                   Diet       0.60      0.49      0.54       229
                              Direction       0.88      0.90      0.89      6187
              Disease_Syndrome_Disorder       0.90      0.89      0.89     13236
                                 Dosage       0.57      0.49      0.53       263
                                   Drug       0.91      0.93      0.92     15926
                               Duration       0.82      0.85      0.83      1218
                           EKG_Findings       0.64      0.70      0.67       325
                             Employment       0.79      0.85      0.82       539
           External_body_part_or_region       0.84      0.84      0.84      4805
                  Family_History_Header       1.00      1.00      1.00       889
                          Fetus_NewBorn       0.57      0.56      0.56       341
                                   Form       0.53      0.43      0.48        81
                              Frequency       0.87      0.90      0.88      1718
                                 Gender       0.98      0.98      0.98      5666
                                    HDL       0.60      1.00      0.75         6
                          Heart_Disease       0.88      0.88      0.88      2295
                                 Height       0.89      0.96      0.92       134
                         Hyperlipidemia       1.00      0.95      0.97       194
                           Hypertension       0.95      0.98      0.97       566
                        ImagingFindings       0.66      0.64      0.65       601
                      Imaging_Technique       0.62      0.67      0.64       108
                    Injury_or_Poisoning       0.85      0.83      0.84      1680
            Internal_organ_or_component       0.90      0.91      0.90     21318
                         Kidney_Disease       0.89      0.89      0.89       446
                                    LDL       0.88      0.97      0.92        37
                        Labour_Delivery       0.82      0.71      0.76       306
                         Medical_Device       0.89      0.93      0.91     12852
                 Medical_History_Header       0.96      0.97      0.96      1013
                               Modifier       0.68      0.60      0.64      1398
                          O2_Saturation       0.84      0.82      0.83       199
                                Obesity       0.96      0.98      0.97       130
                            Oncological       0.88      0.96      0.92      1635
                             Overweight       0.80      0.80      0.80        10
                         Oxygen_Therapy       0.91      0.92      0.92       231
                              Pregnancy       0.81      0.83      0.82       439
                              Procedure       0.91      0.91      0.91     14410
                Psychological_Condition       0.81      0.81      0.81       354
                                  Pulse       0.85      0.95      0.89       389
                         Race_Ethnicity       1.00      1.00      1.00       163
                    Relationship_Status       0.93      0.91      0.92        57
                           RelativeDate       0.83      0.86      0.84      1562
                           RelativeTime       0.74      0.79      0.77       431
                            Respiration       0.99      0.95      0.97       221
                                  Route       0.68      0.69      0.69       597
                         Section_Header       0.97      0.98      0.98     28580
  Sexually_Active_or_Sexual_Orientation       1.00      0.64      0.78        14
                                Smoking       0.83      0.90      0.86       225
                  Social_History_Header       0.95      0.99      0.97       825
                               Strength       0.71      0.55      0.62       227
                              Substance       0.85      0.81      0.83       193
                     Substance_Quantity       0.00      0.00      0.00        28
                                Symptom       0.84      0.86      0.85     23092
                            Temperature       0.94      0.97      0.96       410
                                   Test       0.84      0.88      0.86      9050
                            Test_Result       0.84      0.84      0.84      2766
                                   Time       0.90      0.81      0.86       140
                      Total_Cholesterol       0.69      0.95      0.80        73
                              Treatment       0.73      0.72      0.73       506
                          Triglycerides       0.83      0.80      0.81        30
                             VS_Finding       0.76      0.77      0.76       588
                                Vaccine       0.70      0.84      0.76        92
                     Vital_Signs_Header       0.95      0.98      0.97      2223
                                 Weight       0.88      0.89      0.88       306
                                      O       0.97      0.96      0.97    253164

                               accuracy                           0.94    445974
                              macro avg       0.82      0.82      0.81    445974
                           weighted avg       0.94      0.94      0.94    445974
```

*Example*:

```
documentAssembler = DocumentAssembler()\
       .setInputCol("text")\
       .setOutputCol("document")

sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models")\
       .setInputCols(["document"])\
       .setOutputCol("sentence")

tokenizer = Tokenizer()\
       .setInputCols("sentence")\
       .setOutputCol("token")

tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_ner_jsl", "en", "clinical/models")\
       .setInputCols("token", "sentence")\
       .setOutputCol("ner")\
       .setCaseSensitive(True)
  
ner_converter = NerConverter()\
        .setInputCols(["sentence","token","ner"])\
        .setOutputCol("ner_chunk")

pipeline =  Pipeline(stages=[
       documentAssembler,
       sentenceDetector,
       tokenizer,
       tokenClassifier,
       ner_converter
  ])

p_model = pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

text = 'A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus ( T2DM ), one prior episode of HTG-induced pancreatitis three years prior to presentation , associated with an acute hepatitis , and obesity with a body mass index ( BMI ) of 33.5 kg/m2 , presented with a one-week history of polyuria , polydipsia , poor appetite , and vomiting . Two weeks prior to presentation , she was treated with a five-day course of amoxicillin for a respiratory tract infection . She was on metformin , glipizide , and dapagliflozin for T2DM and atorvastatin and gemfibrozil for HTG . She had been on dapagliflozin for six months at the time of presentation . Physical examination on presentation was significant for dry oral mucosa ; significantly , her abdominal examination was benign with no tenderness , guarding , or rigidity . Pertinent laboratory findings on admission were : serum glucose 111 mg/dl , bicarbonate 18 mmol/l , anion gap 20 , creatinine 0.4 mg/dL , triglycerides 508 mg/dL , total cholesterol 122 mg/dL , glycated hemoglobin ( HbA1c ) 10% , and venous pH 7.27 . Serum lipase was normal at 43 U/L . Serum acetone levels could not be assessed as blood samples kept hemolyzing due to significant lipemia . The patient was initially admitted for starvation ketosis , as she reported poor oral intake for three days prior to admission . However , serum chemistry obtained six hours after presentation revealed her glucose was 186 mg/dL , the anion gap was still elevated at 21 , serum bicarbonate was 16 mmol/L , triglyceride level peaked at 2050 mg/dL , and lipase was 52 U/L . The β-hydroxybutyrate level was obtained and found to be elevated at 5.29 mmol/L - the original sample was centrifuged and the chylomicron layer removed prior to analysis due to interference from turbidity caused by lipemia again . The patient was treated with an insulin drip for euDKA and HTG with a reduction in the anion gap to 13 and triglycerides to 1400 mg/dL , within 24 hours . Her euDKA was thought to be precipitated by her respiratory tract infection in the setting of SGLT2 inhibitor use . The patient was seen by the endocrinology service and she was discharged on 40 units of insulin glargine at night , 12 units of insulin lispro with meals , and metformin 1000 mg two times a day . It was determined that all SGLT2 inhibitors should be discontinued indefinitely . She had close follow-up with endocrinology post discharge .'

res = p_model.transform(spark.createDataFrame([[text]]).toDF("text")).collect()

res[0]['label']
```


#### HCC module added support for versions v22 and v23

Now we can use the version 22 and the version 23 for the new HCC module to calculate CMS-HCC Risk Adjustment score.

Added the following parameters `elig`, `orec` and `medicaid` on the profiles functions. These parameters may not be stored in clinical notes, and may require to be imported from other sources.

```
elig : The eligibility segment of the patient.
       Allowed values are as follows:
       - "CFA": Community Full Benefit Dual Aged
       - "CFD": Community Full Benefit Dual Disabled
       - "CNA": Community NonDual Aged
       - "CND": Community NonDual Disabled
       - "CPA": Community Partial Benefit Dual Aged
       - "CPD": Community Partial Benefit Dual Disabled
       - "INS": Long Term Institutional
       - "NE": New Enrollee
       - "SNPNE": SNP NE
       
orec: Original reason for entitlement code.
      - "0": Old age and survivor's insurance
      - "1": Disability insurance benefits
      - "2": End-stage renal disease 
      - "3": Both DIB and ESRD

medicaid: If the patient is in Medicaid or not.

```

Required parameters should be stored in Spark dataframe.

```python

df.show(truncate=False)

+---------------+------------------------------+---+------+-----------+----+--------+
|hcc_profileV24 |icd10_code                    |age|gender|eligibility|orec|medicaid|
+---------------+------------------------------+---+------+-----------+----+--------+
|{"hcc_lst":[...|[E1169, I5030, I509, E852]    |64 |F     |CFA        |0   |true    |
|{"hcc_lst":[...|[G629, D469, D6181]           |77 |M     |CND        |1   |false   |
|{"hcc_lst":[...|[D473, D473, D473, M069, C969]|16 |F     |CPA        |3   |true    |
+---------------+------------------------------+---+------+-----------+----+--------+

The content of the hcc_profileV24 column is a JSON-parsable string, like in the following example,
{
    "hcc_lst": [
        "HCC18",
        "HCC85_gDiabetesMellit",
        "HCC85",
        "HCC23",
        "D3"
    ],
    "details": {
        "CNA_HCC18": 0.302,
        "CNA_HCC85": 0.331,
        "CNA_HCC23": 0.194,
        "CNA_D3": 0.0,
        "CNA_HCC85_gDiabetesMellit": 0.0
    },
    "hcc_map": {
        "E1169": [
            "HCC18"
        ],
        "I5030": [
            "HCC85"
        ],
        "I509": [
            "HCC85"
        ],
        "E852": [
            "HCC23"
        ]
    },
    "risk_score": 0.827,
    "parameters": {
        "elig": "CNA",
        "age": 56,
        "sex": "F",
        "origds": false,
        "disabled": false,
        "medicaid": false
    }
}


```
We can import different CMS-HCC model versions as seperate functions and use them in the same program.

```python

from sparknlp_jsl.functions import profile,profileV22,profileV23

df = df.withColumn("hcc_profileV24", profile(df.icd10_code, 
                                          df.age, 
                                          df.gender,
                                          df.eligibility,
                                          df.orec,
                                          df.medicaid
                                          ))
                                          
df.withColumn("hcc_profileV22", profileV22(df.codes, df.age, df.sex,df.elig,df.orec,df.medicaid))
df.withColumn("hcc_profileV23", profileV23(df.codes, df.age, df.sex,df.elig,df.orec,df.medicaid))

```


```python
df.show(truncate=False)

+----------+------------------------------+---+------+-----------+----+--------+
|risk_score|icd10_code                    |age|gender|eligibility|orec|medicaid|
+----------+------------------------------+---+------+-----------+----+--------+
|0.922     |[E1169, I5030, I509, E852]    |64 |F     |CFA        |0   |true    |
|3.566     |[G629, D469, D6181]           |77 |M     |CND        |1   |false   |
|1.181     |[D473, D473, D473, M069, C969]|16 |F     |CPA        |3   |true    |
+----------+------------------------------+---+------+-----------+----+--------+
```

#### Updated Notebooks for resolvers and graph builders

+ We have updated the resolver notebooks on spark-nlp-workshop repo with new `BertSentenceChunkEmbeddings` annotator. This annotator lets users aggregate sentence embeddings and ner chunk embeddings to get more specific and accurate resolution codes. It works by averaging context and chunk embeddings to get contextual information. Input to this annotator is the context (sentence) and ner chunks, while the output is embedding for each chunk that can be fed to the resolver model. The `setChunkWeight` parameter can be used to control the influence of surrounding context. Example below shows the comparison of old vs new approach.


|text|ner_chunk|entity|icd10_code|all_codes|resolutions|icd10_code_SCE|all_codes_SCE|resolutions_SCE|
|-|-|-|-|-|-|-|-|-|
|Two weeks prior to presentation, she was treated with a five-day course of amoxicillin for a respiratory tract infection.|a respiratory tract infection|PROBLEM|J988|[J988, J069, A499, J22, J209,...]|[respiratory tract infection, upper respiratory tract infection, bacterial respiratory infection, acute respiratory infection, bronchial infection,...]|Z870|[Z870, Z8709, J470, J988, A499,...|[history of acute lower respiratory tract infection (situation), history of acute lower respiratory tract infection, bronchiectasis with acute lower respiratory infection, rti - respiratory tract infection, bacterial respiratory infection,...|

Here are the updated resolver notebooks:

> - [3.Clinical_Entity_Resolvers.ipynb](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb)
> - [24.Improved_Entity_Resolvers_in_SparkNLP_with_sBert.ipynb](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/24.Improved_Entity_Resolvers_in_SparkNLP_with_sBert.ipynb)

You can also check for more examples of this annotator: [24.1.Improved_Entity_Resolution_with_SentenceChunkEmbeddings.ipynb](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/24.1.Improved_Entity_Resolution_with_SentenceChunkEmbeddings.ipynb)

+ We have updated TF Graph builder notebook to show how to create TF graphs with TF2.x. 

> Here is the updated notebook: [17.Graph_builder_for_DL_models.ipynb](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/17.Graph_builder_for_DL_models.ipynb)  

**To see more, please check: [Spark NLP Healthcare Workshop Repo](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings/Healthcare)**


#### New TF Graph Builder

TF graph builder to create graphs and train DL models for licensed annotators (MedicalNer, Relation Extraction, Assertion and Generic Classifier) is made compatible with TF2.x. 

To see how to create TF Graphs, you can check here: [17.Graph_builder_for_DL_models.ipynb](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/17.Graph_builder_for_DL_models.ipynb)  

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


## 3.1.3
We are glad to announce that Spark NLP for Healthcare 3.1.3 has been released!.
This release comes with new features, new models, bug fixes, and examples.

#### Highlights
+ New Relation Extraction model and a Pretrained pipeline for extracting and linking ADEs
+ New Entity Resolver model for SNOMED codes
+ ChunkConverter Annotator
+ BugFix: getAnchorDateMonth method in DateNormalizer.
+ BugFix: character map in MedicalNerModel fine-tuning.

##### New Relation Extraction model and a Pretrained pipeline for extracting and linking ADEs

We are releasing a new Relation Extraction Model for ADEs. This model is trained using Bert Word embeddings (`biobert_pubmed_base_cased`), and is capable of linking ADEs and Drugs.

*Example*:

```python
re_model = RelationExtractionModel()\
        .pretrained("re_ade_biobert", "en", 'clinical/models')\
        .setInputCols(["embeddings", "pos_tags", "ner_chunks", "dependencies"])\
        .setOutputCol("relations")\
        .setMaxSyntacticDistance(3)\ #default: 0 
        .setPredictionThreshold(0.5)\ #default: 0.5 
        .setRelationPairs(["ade-drug", "drug-ade"]) # Possible relation pairs. Default: All Relations.

nlp_pipeline = Pipeline(stages=[documenter, sentencer, tokenizer, words_embedder, pos_tagger, ner_tagger, ner_chunker, dependency_parser, re_model])

light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))

text ="""Been taking Lipitor for 15 years , have experienced sever fatigue a lot!!! . Doctor moved me to voltaren 2 months ago , so far , have only experienced cramps"""

annotations = light_pipeline.fullAnnotate(text)
```

We also have a new pipeline comprising of all models related to ADE(Adversal Drug Event) as part of this release. This pipeline includes classification, NER, assertion and relation extraction models. Users can now use this pipeline to get classification result, ADE and Drug entities, assertion status for ADE entities, and relations between ADE and Drug entities.

Example:

```python
    pretrained_ade_pipeline = PretrainedPipeline('explain_clinical_doc_ade', 'en', 'clinical/models')
    
    result = pretrained_ade_pipeline.fullAnnotate("""Been taking Lipitor for 15 years , have experienced sever fatigue a lot!!! . Doctor moved me to voltaren 2 months ago , so far , have only experienced cramps""")[0]
```

*Results*:

```bash
Class: True

NER_Assertion:
|    | chunk                   | entitiy    | assertion   |
|----|-------------------------|------------|-------------|
| 0  | Lipitor                 | DRUG       | -           |
| 1  | sever fatigue           | ADE        | Conditional |
| 2  | voltaren                | DRUG       | -           |
| 3  | cramps                  | ADE        | Conditional |

Relations:
|    | chunk1                        | entitiy1   | chunk2      | entity2 | relation |
|----|-------------------------------|------------|-------------|---------|----------|
| 0  | sever fatigue                 | ADE        | Lipitor     | DRUG    |        1 |
| 1  | cramps                        | ADE        | Lipitor     | DRUG    |        0 |
| 2  | sever fatigue                 | ADE        | voltaren    | DRUG    |        0 |
| 3  | cramps                        | ADE        | voltaren    | DRUG    |        1 |

```
##### New Entity Resolver model for SNOMED codes

We are releasing a new SentenceEntityResolver model for SNOMED codes. This model also includes AUX SNOMED concepts and can find codes for Morph Abnormality, Procedure, Substance, Physical Object, and Body Structure entities. In the metadata, the `all_k_aux_labels` can be divided to get further information: `ground truth`, `concept`, and `aux` . In the example shared below the ground truth is `Atherosclerosis`, concept is `Observation`, and aux is `Morph Abnormality`.

*Example*:

```python
snomed_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_snomed_findings_aux_concepts", "en", "clinical/models") \
     .setInputCols(["ner_chunk", "sbert_embeddings"]) \
     .setOutputCol("snomed_code")\
     .setDistanceFunction("EUCLIDEAN")

snomed_pipelineModel = PipelineModel(
    stages = [
        documentAssembler,
        sbert_embedder,
        snomed_resolver])

snomed_lp = LightPipeline(snomed_pipelineModel)
result = snomed_lp.fullAnnotate("atherosclerosis")

```

*Results*:

```bash
|    | chunks          | code     | resolutions                                                                                                                                                                                                                                                                                                                                                                                                                    | all_codes                                                                                                                                                                                          | all_k_aux_labels                                      | all_distances                                                                                                                                   |
|---:|:----------------|:---------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------|
|  0 | atherosclerosis | 38716007 | [atherosclerosis, atherosclerosis, atherosclerosis, atherosclerosis, atherosclerosis, atherosclerosis, atherosclerosis artery, coronary atherosclerosis, coronary atherosclerosis, coronary atherosclerosis, coronary atherosclerosis, coronary atherosclerosis, arteriosclerosis, carotid atherosclerosis, cardiovascular arteriosclerosis, aortic atherosclerosis, aortic atherosclerosis, atherosclerotic ischemic disease] | [38716007, 155382007, 155414001, 195251000, 266318005, 194848007, 441574008, 443502000, 41702007, 266231003, 155316000, 194841001, 28960008, 300920004, 39468009, 155415000, 195252007, 129573006] | 'Atherosclerosis', 'Observation', 'Morph Abnormality' | [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0280, 0.0451, 0.0451, 0.0451, 0.0451, 0.0451, 0.0462, 0.0477, 0.0466, 0.0490, 0.0490, 0.0485 |
```


##### ChunkConverter Annotator

Allows to use RegexMather chunks as NER chunks and feed the output to the downstream annotators like RE or Deidentification.

``` Python
        document_assembler = DocumentAssembler().setInputCol('text').setOutputCol('document')

        sentence_detector = SentenceDetector().setInputCols(["document"]).setOutputCol("sentence")

        regex_matcher = RegexMatcher()\
            .setInputCols("sentence")\
            .setOutputCol("regex")\
            .setExternalRules(path="../src/test/resources/regex-matcher/rules.txt",delimiter=",")

        chunkConverter = ChunkConverter().setInputCols("regex").setOutputCol("chunk")
        

```


## 3.1.2
We are glad to announce that Spark NLP for Healthcare 3.1.2 has been released!.
This release comes with new features, new models, bug fixes, and examples.

#### Highlights
+ Support for Fine-tuning of Ner models.
+ More builtin(pre-defined) graphs for MedicalNerApproach.
+ Date Normalizer.
+ New Relation Extraction Models for ADE.
+ Bug Fixes.
+ Support for user-defined Custom Transformer.
+ Java Workshop Examples.
+ Deprecated Compatibility class in Python.


##### Support for Fine Tuning of Ner models

Users can now resume training/fine-tune existing(already trained) Spark NLP MedicalNer models on new data. Users can simply provide the path to any existing MedicalNer model and train it further on the new dataset:

```
ner_tagger = MedicalNerApproach().setPretrainedModelPath("/path/to/trained/medicalnermodel") 
```

If the new dataset contains new tags/labels/entities, users can choose to override existing tags with the new ones. The default behaviour is to reset the list of existing tags and generate a new list from the new dataset. It is also possible to preserve the existing tags by setting the 'overrideExistingTags' parameter:

```
ner_tagger = MedicalNerApproach()\
  .setPretrainedModelPath("/path/to/trained/medicalnermodel")\
  .setOverrideExistingTags(False)
```

Setting overrideExistingTags to false is intended to be used when resuming trainig on the same, or very similar dataset (i.e. with the same tags or with just a few different ones).

If tags overriding is disabled, and new tags are found in the training set, then the approach will try to allocate them to unused output nodes, if any. It is also possible to override specific tags of the old model by mapping them to new tags:


```bash
ner_tagger = MedicalNerApproach()\
  .setPretrainedModelPath("/path/to/trained/medicalnermodel")\
  .setOverrideExistingTags(False)\
  .setTagsMapping("B-PER,B-VIP", "I-PER,I-VIP")
```

In this case, the new tags `B-VIP` and `I-VIP` will replace the already trained tags 'B-PER' and 'I-PER'. Unmapped old tags will remain in use and unmapped new tags will be allocated to new outpout nodes, if any.

Jupyter Notebook: [Finetuning Medical NER Model Notebook] (https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.5.Resume_MedicalNer_Model_Training.ipynb)

##### More builtin graphs for MedicalNerApproach

Seventy new TensorFlow graphs have been added to the library of available graphs which are used to train MedicalNer models. The graph with the optimal set of parameters is automatically chosen by MedicalNerApproach.


##### DateNormalizer

New annotator that normalize dates to the format YYYY/MM/DD.
This annotator identifies dates in chunk annotations, and transform these dates to the format YYYY/MM/DD.
Both the input and output formats for the annotator are `chunk`.

Example:


```bash
	sentences = [
		    ['08/02/2018'],
		    ['11/2018'],
		    ['11/01/2018'],
		    ['12Mar2021'],
		    ['Jan 30, 2018'],
		    ['13.04.1999'],
		    ['3April 2020'],
		    ['next monday'],
		    ['today'],
		    ['next week'],
	]
	df = spark.createDataFrame(sentences).toDF("text")
	document_assembler = DocumentAssembler().setInputCol('text').setOutputCol('document')
	chunksDF = document_assembler.transform(df)
	aa = map_annotations_col(chunksDF.select("document"),
				    lambda x: [Annotation('chunk', a.begin, a.end, a.result, a.metadata, a.embeddings) for a in x], "document",
				    "chunk_date", "chunk")
	dateNormalizer = DateNormalizer().setInputCols('chunk_date').setOutputCol('date').setAnchorDateYear(2021).setAnchorDateMonth(2).setAnchorDateDay(27)
	dateDf = dateNormalizer.transform(aa)
	dateDf.select("date.result","text").show()
```

```bash
     
+-----------+----------+
|text        |  date    |
+-----------+----------+
|08/02/2018  |2018/08/02|
|11/2018     |2018/11/DD|
|11/01/2018  |2018/11/01|
|12Mar2021   |2021/03/12|
|Jan 30, 2018|2018/01/30|
|13.04.1999  |1999/04/13|
|3April 2020 |2020/04/03|
|next Monday |2021/06/19|
|today       |2021/06/12|
|next week   |2021/06/19|
+-----------+----------+
```

##### New Relation Extraction Models for ADE

We are releasing new Relation Extraction models for ADE (Adverse Drug Event). This model is available in both `RelationExtraction` and Bert based `RelationExtractionDL` versions, and is capabale of linking drugs with ADE mentions.

Example

```bash
    ade_re_model = new RelationExtractionModel().pretrained('ner_ade_clinical', 'en', 'clinical/models')\
                                     .setInputCols(["embeddings", "pos_tags", "ner_chunk", "dependencies"])\
                                     .setOutputCol("relations")\
                                     .setPredictionThreshold(0.5)\
                                     .setRelationPairs(['ade-drug', 'drug-ade'])
    pipeline = Pipeline(stages=[documenter, sentencer, tokenizer, pos_tagger, words_embedder, ner_tagger, ner_converter, 
                                                   dependency_parser, re_ner_chunk_filter, re_model])
    text ="""A 30 year old female presented with tense bullae due to excessive use of naproxin, and leg cramps relating to oxaprozin."""

    p_model = pipeline.fit(spark.createDataFrame([[text]]).toDF("text"))
    
    result = p_model.transform(data)
       
```

Results
```bash
|    | chunk1        | entity1    |  chunk2       | entity2   |    result  |
|---:|:--------------|:-----------|:--------------|:----------|-----------:|
|  0 | tense bullae  | ADE        | naproxin      | DRUG      |          1 |
|  1 | tense bullae  | ADE        | oxaprozin     | DRUG      |          0 |
|  2 | naproxin      | DRUG       | leg cramps    | ADE       |          0 |
|  3 | leg cramps    | ADE        | oxaprozin     | DRUG      |          1 |

```

Benchmarking
Model: `re_ade_clinical`
```bash

              precision    recall  f1-score   support
           0       0.85      0.89      0.87      1670
           1       0.88      0.84      0.86      1673
   micro avg       0.87      0.87      0.87      3343
   macro avg       0.87      0.87      0.87      3343
weighted avg       0.87      0.87      0.87      3343
```

Model: `redl_ade_biobert`
```bash
Relation           Recall Precision        F1   Support
0                   0.894     0.946     0.919      1011
1                   0.963     0.926     0.944      1389
Avg.                0.928     0.936     0.932
Weighted Avg.       0.934     0.934     0.933
```

##### Bug Fixes
+ RelationExtractionDLModel had an issue(BufferOverflowException) on versions 3.1.0 and 3.1.1, which is fixed with this release.
+ Some pretrained RelationExtractionDLModels got outdated after release 3.0.3, new updated models were created, tested and made available to be used with versions 3.0.3, and later.
+ Some SentenceEntityResolverModels which did not work with Spark 2.4/2.3 were fixed.

##### Support for user-defined Custom Transformer.
Utility classes to define custom transformers in python are included in this release. This allows users to define functions in Python to manipulate Spark-NLP annotations. This new Transformers can be added to pipelines like any of the other models you're already familiar with.
Example how to use the custom transformer.

```python
        def myFunction(annotations):
            # lower case the content of the annotations
            return [a.copy(a.result.lower()) for a in annotations]

        custom_transformer = CustomTransformer(f=myFunction).setInputCol("ner_chunk").setOutputCol("custom")
        outputDf = custom_transformer.transform(outdf).select("custom").toPandas()
```

##### Java Workshop Examples

Add Java examples in the workshop repository.
https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/java/healthcare

##### Deprecated Compatibility class in Python

Due to active release cycle, we are adding & training new pretrained models at each release and it might be tricky to maintain the backward compatibility or keep up with the latest models and versions, especially for the users using our models locally in air-gapped networks.

We are releasing a new utility class to help you check your local & existing models with the latest version of everything we have up to date. You will not need to specify your AWS credentials from now on. This new class is now replacing the previous Compatibility class written in Python and CompatibilityBeta class written in Scala.

```        
from sparknlp_jsl.compatibility import Compatibility

compatibility = Compatibility(spark)

print(compatibility.findVersion('sentence_detector_dl_healthcare'))
```

Output
```
[{'name': 'sentence_detector_dl_healthcare', 'sparkVersion': '2.4', 'version': '2.6.0', 'language': 'en', 'date': '2020-09-13T14:44:42.565', 'readyToUse': 'true'}, {'name': 'sentence_detector_dl_healthcare', 'sparkVersion': '2.4', 'version': '2.7.0', 'language': 'en', 'date': '2021-03-16T08:42:34.391', 'readyToUse': 'true'}]
```


## 3.1.1
We are glad to announce that Spark NLP for Healthcare 3.1.1 has been released!

#### Highlights
+ MedicalNerModel new parameter `includeAllConfidenceScores`.
+ MedicalNerModel new parameter `inferenceBatchSize`.
+ New Resolver Models
+ Updated Resolver Models
+ Getting Started with Spark NLP for Healthcare Notebook in Databricks

#### MedicalNer new parameter `includeAllConfidenceScores`
You can now customize whether you will require confidence score for every token(both entities and non-entities) at the output of the MedicalNerModel, or just for the tokens recognized as entities.

#### MedicalNerModel new parameter `inferenceBatchSize`
You can now control the batch size used during inference as a separate parameter from the one you used during training of the model. This can be useful in the situation in which the hardware on which you run inference has different capacity. For example, when you have lower available memory during inference, you can reduce the batch size.
 
#### New Resolver Models
We trained three new sentence entity resolver models. 

+ `sbertresolve_snomed_bodyStructure_med` and `sbiobertresolve_snomed_bodyStructure` models map extracted medical (anatomical structures) entities to Snomed codes (body structure version). 

     + `sbertresolve_snomed_bodyStructure_med` : Trained  with using `sbert_jsl_medium_uncased` embeddings.
     + `sbiobertresolve_snomed_bodyStructure`  : Trained with using `sbiobert_base_cased_mli` embeddings.

*Example* :
```
documentAssembler = DocumentAssembler()\
      .setInputCol("text")\
      .setOutputCol("ner_chunk")
jsl_sbert_embedder = BertSentenceEmbeddings.pretrained('sbert_jsl_medium_uncased','en','clinical/models')\
      .setInputCols(["ner_chunk"])\
      .setOutputCol("sbert_embeddings")
snomed_resolver = SentenceEntityResolverModel.pretrained("sbertresolve_snomed_bodyStructure_med, "en", "clinical/models) \
      .setInputCols(["ner_chunk", "sbert_embeddings"]) \
      .setOutputCol("snomed_code")
snomed_pipelineModel = PipelineModel(
    stages = [
        documentAssembler,
        jsl_sbert_embedder,
        snomed_resolver])
snomed_lp = LightPipeline(snomed_pipelineModel)
result = snomed_lp.fullAnnotate("Amputation stump")
```

*Result*:
```bash
|    | chunks           | code     | resolutions                                                                                                                                                                                                                                  | all_codes                                                                                       | all_distances                                                               |
|---:|:-----------------|:---------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------|
|  0 | amputation stump | 38033009 | [Amputation stump, Amputation stump of upper limb, Amputation stump of left upper limb, Amputation stump of lower limb, Amputation stump of left lower limb, Amputation stump of right upper limb, Amputation stump of right lower limb, ...]| ['38033009', '771359009', '771364008', '771358001', '771367001', '771365009', '771368006', ...] | ['0.0000', '0.0773', '0.0858', '0.0863', '0.0905', '0.0911', '0.0972', ...] |
```
 +  `sbiobertresolve_icdo_augmented` : This model maps extracted medical entities to ICD-O codes using sBioBert sentence embeddings. This model is augmented using the site information coming from ICD10 and synonyms coming from SNOMED vocabularies. It is trained with a dataset that is 20x larger than the previous version of ICDO resolver. Given the oncological entity found in the text (via NER models like ner_jsl), it returns top terms and resolutions along with the corresponding ICD-10 codes to present more granularity with respect to body parts mentioned. It also returns the original histological behavioral codes and descriptions in the aux metadata.

*Example*:

```
...
chunk2doc = Chunk2Doc().setInputCols("ner_chunk").setOutputCol("ner_chunk_doc")
 
sbert_embedder = BertSentenceEmbeddings\
     .pretrained("sbiobert_base_cased_mli","en","clinical/models")\
     .setInputCols(["ner_chunk_doc"])\
     .setOutputCol("sbert_embeddings")
 
icdo_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_icdo_augmented","en", "clinical/models") \
     .setInputCols(["ner_chunk", "sbert_embeddings"]) \
     .setOutputCol("resolution")\
     .setDistanceFunction("EUCLIDEAN")
nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter, chunk2doc, sbert_embedder, icdo_resolver])
empty_data = spark.createDataFrame([[""]]).toDF("text")
model = nlpPipeline.fit(empty_data)
results = model.transform(spark.createDataFrame([["The patient is a very pleasant 61-year-old female with a strong family history of colon polyps. The patient reports her first polyps noted at the age of 50. We reviewed the pathology obtained from the pericardectomy in March 2006, which was diagnostic of mesothelioma. She also has history of several malignancies in the family. Her father died of a brain tumor at the age of 81. Her sister died at the age of 65 breast cancer. She has two maternal aunts with history of lung cancer both of whom were smoker. Also a paternal grandmother who was diagnosed with leukemia at 86 and a paternal grandfather who had B-cell lymphoma."]]).toDF("text"))
```

*Result*:

```
+--------------------+-----+---+-----------+-------------+-------------------------+-------------------------+
|               chunk|begin|end|     entity|         code|        all_k_resolutions|              all_k_codes|
+--------------------+-----+---+-----------+-------------+-------------------------+-------------------------+
|        mesothelioma|  255|266|Oncological|9971/3||C38.3|malignant mediastinal ...|9971/3||C38.3:::8854/3...|
|several malignancies|  293|312|Oncological|8894/3||C39.8|overlapping malignant ...|8894/3||C39.8:::8070/2...|
|         brain tumor|  350|360|Oncological|9562/0||C71.9|cancer of the brain:::...|9562/0||C71.9:::9070/3...|
|       breast cancer|  413|425|Oncological|9691/3||C50.9|carcinoma of breast:::...|9691/3||C50.9:::8070/2...|
|         lung cancer|  471|481|Oncological|8814/3||C34.9|malignant tumour of lu...|8814/3||C34.9:::8550/3...|
|            leukemia|  560|567|Oncological|9670/3||C80.9|anemia in neoplastic d...|9670/3||C80.9:::9714/3...|
|     B-cell lymphoma|  610|624|Oncological|9818/3||C77.9|secondary malignant ne...|9818/3||C77.9:::9655/3...|
+--------------------+-----+---+-----------+-------------+-------------------------+-------------------------+
```

 
#### Updated Resolver Models
We updated `sbiobertresolve_snomed_findings` and `sbiobertresolve_cpt_procedures_augmented` resolver models to reflect the latest changes in the official terminologies.

#### Getting Started with Spark NLP for Healthcare Notebook in Databricks
We prepared a new notebook for those who want to get started with Spark NLP for Healthcare in Databricks : [Getting Started with Spark NLP for Healthcare Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/databricks/python/healthcare_case_studies/Get_Started_Spark_NLP_for_Healthcare.ipynb) 

## 3.1.0
We are glad to announce that Spark NLP for Healthcare 3.1.0 has been released!
#### Highlights
+ Improved load time & memory consumption for SentenceResolver models.
+ New JSL Bert Models.
+ JSL SBert Model Speed Benchmark.
+ New ICD10CM resolver models.
+ New Deidentification NER models.
+ New column returned in DeidentificationModel
+ New Reidentification feature
+ New Deidentification Pretrained Pipelines
+ Chunk filtering based on confidence
+ Extended regex dictionary fuctionallity in Deidentification
+ Enhanced RelationExtractionDL Model to create and identify relations between entities across the entire document
+ MedicalNerApproach can now accept a graph file directly.
+ MedicalNerApproach can now accept a user-defined name for log file.
+ More improvements in Scaladocs. 
+ Bug fixes in Deidentification module.
+ New notebooks.



#### Sentence Resolver Models load time improvement
Sentence resolver models now have faster load times, with a speedup of about 6X when compared to previous versions.
Also, the load process now is more  memory friendly meaning that the maximum memory required during load time is smaller, reducing the chances of OOM exceptions, and thus relaxing hardware requirements.

#### New JSL SBert Models
We trained new sBert models in TF2 and fined tuned on MedNLI, NLI and UMLS datasets with various parameters to cover common NLP tasks in medical domain. You can find the details in the following table.

+ `sbiobert_jsl_cased`
+ `sbiobert_jsl_umls_cased`
+ `sbert_jsl_medium_uncased`
+ `sbert_jsl_medium_umls_uncased`
+ `sbert_jsl_mini_uncased`
+ `sbert_jsl_mini_umls_uncased`
+ `sbert_jsl_tiny_uncased`
+ `sbert_jsl_tiny_umls_uncased`

#### JSL SBert Model Speed Benchmark

| JSL SBert Model| Base Model | Is Cased | Train Datasets | Inference speed (100 rows) |
|-|-|-|-|-|
| sbiobert_jsl_cased | biobert_v1.1_pubmed | Cased | medNLI, allNLI| 274,53 |
| sbiobert_jsl_umls_cased | biobert_v1.1_pubmed | Cased | medNLI, allNLI, umls | 274,52 |
| sbert_jsl_medium_uncased | uncased_L-8_H-512_A-8 | Uncased | medNLI, allNLI| 80,40 |
| sbert_jsl_medium_umls_uncased | uncased_L-8_H-512_A-8 | Uncased | medNLI, allNLI, umls | 78,35 |
| sbert_jsl_mini_uncased | uncased_L-4_H-256_A-4 | Uncased | medNLI, allNLI| 10,68 |
| sbert_jsl_mini_umls_uncased | uncased_L-4_H-256_A-4 | Uncased | medNLI, allNLI, umls | 10,29 |
| sbert_jsl_tiny_uncased | uncased_L-2_H-128_A-2 | Uncased | medNLI, allNLI| 4,54 |
| sbert_jsl_tiny_umls_uncased | uncased_L-2_H-128_A-2 | Uncased | medNLI, allNL, umls | 4,54 |

#### New ICD10CM resolver models:
These models map clinical entities and concepts to ICD10 CM codes using sentence bert embeddings. They also return the official resolution text within the brackets inside the metadata. Both models are augmented with synonyms, and previous augmentations are flexed according to cosine distances to unnormalized terms (ground truths).

+ `sbiobertresolve_icd10cm_slim_billable_hcc`: Trained with classic sbiobert mli. (`sbiobert_base_cased_mli`)

Models Hub Page : https://nlp.johnsnowlabs.com/2021/05/25/sbiobertresolve_icd10cm_slim_billable_hcc_en.html

+ `sbertresolve_icd10cm_slim_billable_hcc_med`: Trained with new jsl sbert(`sbert_jsl_medium_uncased`)

Models Hub Page : https://nlp.johnsnowlabs.com/2021/05/25/sbertresolve_icd10cm_slim_billable_hcc_med_en.html


*Example*: 'bladder cancer' 
+ `sbiobertresolve_icd10cm_augmented_billable_hcc`

| chunks | code | all_codes | resolutions |all_distances | 100x Loop(sec) |
|-|-|-|-|-|-|
| bladder cancer | C679 | [C679, Z126, D090, D494, C7911] | [bladder cancer, suspected bladder cancer, cancer in situ of urinary bladder, tumor of bladder neck, malignant tumour of bladder neck] | [0.0000, 0.0904, 0.0978, 0.1080, 0.1281] | 26,9 |

+ ` sbiobertresolve_icd10cm_slim_billable_hcc`

| chunks | code | all_codes | resolutions |all_distances | 100x Loop(sec) |
| - | - | - | - | - | - |
| bladder cancer | D090 | [D090, D494, C7911, C680, C679] | [cancer in situ of urinary bladder [Carcinoma in situ of bladder], tumor of bladder neck [Neoplasm of unspecified behavior of bladder], malignant tumour of bladder neck [Secondary malignant neoplasm of bladder], carcinoma of urethra [Malignant neoplasm of urethra], malignant tumor of urinary bladder [Malignant neoplasm of bladder, unspecified]] | [0.0978, 0.1080, 0.1281, 0.1314, 0.1284] | 20,9 |

+ `sbertresolve_icd10cm_slim_billable_hcc_med`

| chunks | code | all_codes | resolutions |all_distances | 100x Loop(sec) |
| - | - | - | - | - | - |
| bladder cancer | C671 | [C671, C679, C61, C672, C673] | [bladder cancer, dome [Malignant neoplasm of dome of bladder], cancer of the urinary bladder [Malignant neoplasm of bladder, unspecified], prostate cancer [Malignant neoplasm of prostate], cancer of the urinary bladder] | [0.0894, 0.1051, 0.1184, 0.1180, 0.1200] | 12,8 | 

#### New Deidentification NER Models

We trained four new NER models to find PHI data (protected health information) that may need to be deidentified. `ner_deid_generic_augmented` and `ner_deid_subentity_augmented` models are trained with a combination of 2014 i2b2 Deid dataset and in-house annotations as well as some augmented version of them. Compared to the same test set coming from 2014 i2b2 Deid dataset, we achieved a better accuracy and generalisation on some entity labels as summarised in the following tables. We also trained the same models with `glove_100d` embeddings to get more memory friendly versions.    

+ `ner_deid_generic_augmented`  : Detects PHI 7 entities (`DATE`, `NAME`, `LOCATION`, `PROFESSION`, `CONTACT`, `AGE`, `ID`).

Models Hub Page : https://nlp.johnsnowlabs.com/2021/06/01/ner_deid_generic_augmented_en.html

| entity| ner_deid_large (v3.0.3 and before)|ner_deid_generic_augmented (v3.1.0)|
|--|:--:|:--:|
|   CONTACT| 0.8695|0.9592 |
|      NAME| 0.9452|0.9648 |
|      DATE| 0.9778|0.9855 |
|  LOCATION| 0.8755|0.923 |

+ `ner_deid_subentity_augmented`: Detects PHI 23 entities (`MEDICALRECORD`, `ORGANIZATION`, `DOCTOR`, `USERNAME`, `PROFESSION`, `HEALTHPLAN`, `URL`, `CITY`, `DATE`, `LOCATION-OTHER`, `STATE`, `PATIENT`, `DEVICE`, `COUNTRY`, `ZIP`, `PHONE`, `HOSPITAL`, `EMAIL`, `IDNUM`, `SREET`, `BIOID`, `FAX`, `AGE`)

Models Hub Page : https://nlp.johnsnowlabs.com/2021/06/01/ner_deid_subentity_augmented_en.html

|entity| ner_deid_enriched (v3.0.3 and before)| ner_deid_subentity_augmented (v3.1.0)|
|-------------|:------:|:--------:|
|     HOSPITAL| 0.8519| 0.8983 |
|         DATE| 0.9766| 0.9854 |
|         CITY| 0.7493| 0.8075 |
|       STREET| 0.8902| 0.9772 |
|          ZIP| 0.8| 0.9504 |
|        PHONE| 0.8615| 0.9502 |
|       DOCTOR| 0.9191| 0.9347 |
|          AGE| 0.9416| 0.9469 |

+ `ner_deid_generic_glove`: Small version of `ner_deid_generic_augmented` and detects 7 entities.
+ `ner_deid_subentity_glove`: Small version of `ner_deid_subentity_augmented` and detects 23 entities.

*Example*:

Scala
```scala
...
val deid_ner = MedicalNerModel.pretrained("ner_deid_subentity_augmented", "en", "clinical/models") \
      .setInputCols(Array("sentence", "token", "embeddings")) \
      .setOutputCol("ner")
...
val nlpPipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, deid_ner, ner_converter))
model = nlpPipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

val result = pipeline.fit(Seq.empty["A. Record date : 2093-01-13, David Hale, M.D., Name : Hendrickson, Ora MR. # 7194334 Date : 01/13/93 PCP : Oliveira, 25 -year-old, Record date : 1-11-2000. Cocke County Baptist Hospital. 0295 Keats Street. Phone +1 (302) 786-5227."].toDS.toDF("text")).transform(data)
```

Python
```bash
...
deid_ner = MedicalNerModel.pretrained("ner_deid_subentity_augmented", "en", "clinical/models") \
      .setInputCols(["sentence", "token", "embeddings"]) \
      .setOutputCol("ner")
...
nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, deid_ner, ner_converter])
model = nlpPipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

results = model.transform(spark.createDataFrame(pd.DataFrame({"text": ["""A. Record date : 2093-01-13, David Hale, M.D., Name : Hendrickson, Ora MR. # 7194334 Date : 01/13/93 PCP : Oliveira, 25 -year-old, Record date : 1-11-2000. Cocke County Baptist Hospital. 0295 Keats Street. Phone +1 (302) 786-5227."""]})))
```
*Results*:
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
|25-year-old                  |AGE          |
|1-11-2000                    |DATE         |
|Cocke County Baptist Hospital|HOSPITAL     |
|0295 Keats Street.           |STREET       |
|(302) 786-5227               |PHONE        |
|Brothers Coal-Mine           |ORGANIZATION |
+-----------------------------+-------------+
``` 


#### New column returned in DeidentificationModel

DeidentificationModel now can return a new column to save the mappings between the mask/obfuscated entities and original entities.
This column is optional and you can set it up with the `.setReturnEntityMappings(True)` method. The default value is False.
Also, the name for the column can be changed using the following method; `.setMappingsColumn("newAlternativeName")`
The new column will produce annotations with the following structure,
```
Annotation(
  type: chunk, 
  begin: 17, 
  end: 25, 
  result: 47,
    metadata:{
        originalChunk -> 01/13/93  //Original text of the chunk
        chunk -> 0  // The number of the chunk in the sentence
        beginOriginalChunk -> 95 // Start index of the original chunk
        endOriginalChunk -> 102  // End index of the original chunk
        entity -> AGE // Entity of the chunk
        sentence -> 2 // Number of the sentence
    }
)
```

#### New Reidentification feature

With the new ReidetificationModel, the user can go back to the original sentences using the mappings columns and the deidentification sentences.

*Example:*

Scala

```scala
val redeidentification = new ReIdentification()
     .setInputCols(Array("mappings", "deid_chunks"))
     .setOutputCol("original")
```

Python

```scala
reDeidentification = ReIdentification()
     .setInputCols(["mappings","deid_chunks"])
     .setOutputCol("original")
```

#### New Deidentification Pretrained Pipelines
We developed a `clinical_deidentification` pretrained pipeline that can be used to deidentify PHI information from medical texts. The PHI information will be masked and obfuscated in the resulting text. The pipeline can mask and obfuscate `AGE`, `CONTACT`, `DATE`, `ID`, `LOCATION`, `NAME`, `PROFESSION`, `CITY`, `COUNTRY`, `DOCTOR`, `HOSPITAL`, `IDNUM`, `MEDICALRECORD`, `ORGANIZATION`, `PATIENT`, `PHONE`, `PROFESSION`,  `STREET`, `USERNAME`, `ZIP`, `ACCOUNT`, `LICENSE`, `VIN`, `SSN`, `DLN`, `PLATE`, `IPADDR` entities. 

Models Hub Page : [clinical_deidentification](https://nlp.johnsnowlabs.com/2021/05/27/clinical_deidentification_en.html)

There is also a lightweight version of the same pipeline trained with memory efficient `glove_100d`embeddings.
Here are the model names:
- `clinical_deidentification`
- `clinical_deidentification_glove`

*Example:*

Python:
```bash 
from sparknlp.pretrained import PretrainedPipeline
deid_pipeline = PretrainedPipeline("clinical_deidentification", "en", "clinical/models")

deid_pipeline.annotate("Record date : 2093-01-13, David Hale, M.D. IP: 203.120.223.13. The driver's license no:A334455B. the SSN:324598674 and e-mail: hale@gmail.com. Name : Hendrickson, Ora MR. # 719435 Date : 01/13/93. PCP : Oliveira, 25 years-old. Record date : 2079-11-09, Patient's VIN : 1HGBH41JXMN109286.")
```
Scala:
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
val deid_pipeline = PretrainedPipeline("clinical_deidentification","en","clinical/models")

val result = deid_pipeline.annotate("Record date : 2093-01-13, David Hale, M.D. IP: 203.120.223.13. The driver's license no:A334455B. the SSN:324598674 and e-mail: hale@gmail.com. Name : Hendrickson, Ora MR. # 719435 Date : 01/13/93. PCP : Oliveira, 25 years-old. Record date : 2079-11-09, Patient's VIN : 1HGBH41JXMN109286.")
```
*Result*:
```bash
{'sentence': ['Record date : 2093-01-13, David Hale, M.D.',
   'IP: 203.120.223.13.',
   'The driver's license no:A334455B.',
   'the SSN:324598674 and e-mail: hale@gmail.com.',
   'Name : Hendrickson, Ora MR. # 719435 Date : 01/13/93.',
   'PCP : Oliveira, 25 years-old.',
   'Record date : 2079-11-09, Patient's VIN : 1HGBH41JXMN109286.'],
'masked': ['Record date : <DATE>, <DOCTOR>, M.D.',
   'IP: <IPADDR>.',
   'The driver's license <DLN>.',
   'the <SSN> and e-mail: <EMAIL>.',
   'Name : <PATIENT> MR. # <MEDICALRECORD> Date : <DATE>.',
   'PCP : <DOCTOR>, <AGE> years-old.',
   'Record date : <DATE>, Patient's VIN : <VIN>.'],
'obfuscated': ['Record date : 2093-01-18, Dr Alveria Eden, M.D.',
   'IP: 001.001.001.001.',
   'The driver's license K783518004444.',
   'the SSN-400-50-8849 and e-mail: Merilynn@hotmail.com.',
   'Name : Charls Danger MR. # J3366417 Date : 01-18-1974.',
   'PCP : Dr Sina Sewer, 55 years-old.',
   'Record date : 2079-11-23, Patient's VIN : 6ffff55gggg666777.'],
'ner_chunk': ['2093-01-13',
   'David Hale',
   'no:A334455B',
   'SSN:324598674',
   'Hendrickson, Ora',
   '719435',
   '01/13/93',
   'Oliveira',
   '25',
   '2079-11-09',
   '1HGBH41JXMN109286']}
```

#### Chunk filtering based on confidence

We added a new annotator ChunkFiltererApproach that allows to load a csv with both entities and confidence thresholds. 
This annotator will produce a ChunkFilterer model.

You can load the dictionary with the following property `setEntitiesConfidenceResource()`.

An example dictionary is:

```CSV
TREATMENT,0.7
```

With that dictionary, the user can filter the chunks corresponding to treatment entities which have confidence lower than 0.7.

Example:

We have a ner_chunk column and sentence column with the following data:

Ner_chunk
```
|[{chunk, 141, 163, the genomicorganization, {entity -> TREATMENT, sentence -> 0, chunk -> 0, confidence -> 0.57785}, []}, {chunk, 209, 267, a candidate gene forType II 
           diabetes mellitus, {entity -> PROBLEM, sentence -> 0, chunk -> 1, confidence -> 0.6614286}, []}, {chunk, 394, 408, byapproximately, {entity -> TREATMENT, sentence -> 1, chunk -> 2, confidence -> 0.7705}, []}, {chunk, 478, 508, single nucleotide polymorphisms, {entity -> TREATMENT, sentence -> 2, chunk -> 3, confidence -> 0.7204666}, []}, {chunk, 559, 581, aVal366Ala substitution, {entity -> TREATMENT, sentence -> 2, chunk -> 4, confidence -> 0.61505}, []}, {chunk, 588, 601, an 8 base-pair, {entity -> TREATMENT, sentence -> 2, chunk -> 5, confidence -> 0.29226667}, []}, {chunk, 608, 625, insertion/deletion, {entity -> PROBLEM, sentence -> 3, chunk -> 6, confidence -> 0.9841}, []}]|
+-------
```
Sentence
```
[{document, 0, 298, The human KCNJ9 (Kir 3.3, GIRK3) is a member of the G-protein-activated inwardly rectifying potassium (GIRK) channel family.Here we describe the genomicorganization of the KCNJ9 locus on chromosome 1q21-23 as a candidate gene forType II 
             diabetes mellitus in the Pima Indian population., {sentence -> 0}, []}, {document, 300, 460, The gene spansapproximately 7.6 kb and contains one noncoding and two coding exons ,separated byapproximately 2.2 and approximately 2.6 kb introns, respectively., {sentence -> 1}, []}, {document, 462, 601, We identified14 single nucleotide polymorphisms (SNPs),
             including one that predicts aVal366Ala substitution, and an 8 base-pair, {sentence -> 2}, []}, {document, 603, 626, (bp) insertion/deletion., {sentence -> 3}, []}]
```

We can filter the entities using the following annotator:

```python
        chunker_filter = ChunkFiltererApproach().setInputCols("sentence", "ner_chunk") \
            .setOutputCol("filtered") \
            .setCriteria("regex") \
            .setRegex([".*"]) \         
            .setEntitiesConfidenceResource("entities_confidence.csv")

```
Where entities-confidence.csv has the following data:

```csv
TREATMENT,0.7
PROBLEM,0.9
```
We can use that chunk_filter:

```
chunker_filter.fit(data).transform(data)
```
Producing the following entities:

```
|[{chunk, 394, 408, byapproximately, {entity -> TREATMENT, sentence -> 1, chunk -> 2, confidence -> 0.7705}, []}, {chunk, 478, 508, single nucleotide polymorphisms, {entity -> TREATMENT, sentence -> 2, chunk -> 3, confidence -> 0.7204666}, []}, {chunk, 608, 625, insertion/deletion, {entity -> PROBLEM, sentence -> 3, chunk -> 6, confidence -> 0.9841}, []}]|

```
As you can see, only the treatment entities with confidence score of more than 0.7, and the problem entities with confidence score of more than 0.9 have been kept in the output.


#### Extended regex dictionary fuctionallity in Deidentification

The RegexPatternsDictionary can now use a regex that spawns the 2 previous token and the 2 next tokens.
That feature is implemented using regex groups.

Examples: 

Given the sentence `The patient with ssn 123123123` we can use the following regex to capture the entitty `ssn (\d{9})`
Given the sentence `The patient has 12 years` we can use the following regex to capture the entitty `(\d{2}) years`

#### Enhanced RelationExtractionDL Model to create and identify relations between entities across the entire document

A new option has been added to `RENerChunksFilter` to support pairing entities from different sentences using `.setDocLevelRelations(True)`, to pass to the Relation Extraction Model. The RelationExtractionDL Model has also been updated to process document-level relations.

How to use:

```python
        re_dl_chunks = RENerChunksFilter() \
            .setInputCols(["ner_chunks", "dependencies"])\
            .setDocLevelRelations(True)\
            .setMaxSyntacticDistance(7)\
            .setOutputCol("redl_ner_chunks")  
```

Examples:

Given a document containing multiple sentences: `John somkes cigrettes. He also consumes alcohol.`, now we can generate relation pairs across sentences and relate `alcohol` with `John` .

#### Set NER graph explicitely in MedicalNerApproach
Now MedicalNerApproach can receives the path to the graph directly. When a graph location is provided through this method, previous graph search behavior is disabled.
```
MedicalNerApproach.setGraphFile(graphFilePath)
```

#### MedicalNerApproach can now accept a user-defined name for log file.
Now MedicalNerApproach can accept a user-defined name for the log file. If not such a name is provided, the conventional naming will take place.
```
MedicalNerApproach.setLogPrefix("oncology_ner")
```
This will result in `oncology_ner_20210605_141701.log` filename being used, in which the `20210605_141701` is a timestamp.

#### New Notebooks

+ A [new notebook](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.4.Biomedical_NER_SparkNLP_paper_reproduce.ipynb)  to reproduce our peer-reviewed NER paper (https://arxiv.org/abs/2011.06315)
+ New databricks [case study notebooks](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/databricks/python/healthcare_case_studies). In these notebooks, we showed the examples of how to work with oncology notes dataset and OCR on databricks for both DBr and community edition versions.


### 3.0.3

We are glad to announce that Spark NLP for Healthcare 3.0.3 has been released!

#### Highlights

+ Five new entity resolution models to cover UMLS, HPO and LIONC terminologies.
+ New feature for random displacement of dates on deidentification model.
+ Five new pretrained pipelines to map terminologies across each other (from UMLS to ICD10, from RxNorm to MeSH etc.)
+ AnnotationToolReader support for Spark 2.3. The tool that helps model training on Spark-NLP to leverage data annotated using JSL Annotation Tool now has support for Spark 2.3.
+ Updated documentation (Scaladocs) covering more APIs, and examples. 

#### Five new resolver models:

+ `sbiobertresolve_umls_major_concepts`: This model returns CUI (concept unique identifier) codes for Clinical Findings, Medical Devices, Anatomical Structures and Injuries & Poisoning terms.            
+ `sbiobertresolve_umls_findings`: This model returns CUI (concept unique identifier) codes for 200K concepts from clinical findings.
+ `sbiobertresolve_loinc`: Map clinical NER entities to LOINC codes using `sbiobert`.
+ `sbluebertresolve_loinc`: Map clinical NER entities to LOINC codes using `sbluebert`.
+ `sbiobertresolve_HPO`: This model returns Human Phenotype Ontology (HPO) codes for phenotypic abnormalities encountered in human diseases. It also returns associated codes from the following vocabularies for each HPO code:

		* MeSH (Medical Subject Headings)
		* SNOMED
		* UMLS (Unified Medical Language System )
		* ORPHA (international reference resource for information on rare diseases and orphan drugs)
		* OMIM (Online Mendelian Inheritance in Man)

  *Related Notebook*: [Resolver Models](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/24.Improved_Entity_Resolvers_in_SparkNLP_with_sBert.ipynb)
  
#### New feature on Deidentification Module
+ isRandomDateDisplacement(True): Be able to apply a random displacement on obfuscation dates. The randomness is based on the seed.
+ Fix random dates when the format is not correct. Now you can repeat an execution using a seed for dates. Random dates will be based on the seed. 

#### Five new healthcare code mapping pipelines:

+ `icd10cm_umls_mapping`: This pretrained pipeline maps ICD10CM codes to UMLS codes without using any text data. You’ll just feed white space-delimited ICD10CM codes and it will return the corresponding UMLS codes as a list. If there is no mapping, the original code is returned with no mapping.

      {'icd10cm': ['M89.50', 'R82.2', 'R09.01'], 
          'umls': ['C4721411', 'C0159076', 'C0004044']}

+ `mesh_umls_mapping`: This pretrained pipeline maps MeSH codes to UMLS codes without using any text data. You’ll just feed white space-delimited MeSH codes and it will return the corresponding UMLS codes as a list. If there is no mapping, the original code is returned with no mapping.

      {'mesh': ['C028491', 'D019326', 'C579867'],
         'umls': ['C0970275', 'C0886627', 'C3696376']}

+ `rxnorm_umls_mapping`: This pretrained pipeline maps RxNorm codes to UMLS codes without using any text data. You’ll just feed white space-delimited RxNorm codes and it will return the corresponding UMLS codes as a list. If there is no mapping, the original code is returned with no mapping.

      {'rxnorm': ['1161611', '315677', '343663'],
         'umls': ['C3215948', 'C0984912', 'C1146501']}

+ `rxnorm_mesh_mapping`: This pretrained pipeline maps RxNorm codes to MeSH codes without using any text data. You’ll just feed white space-delimited RxNorm codes and it will return the corresponding MeSH codes as a list. If there is no mapping, the original code is returned with no mapping.

      {'rxnorm': ['1191', '6809', '47613'],
         'mesh': ['D001241', 'D008687', 'D019355']}

+ `snomed_umls_mapping`: This pretrained pipeline maps SNOMED codes to UMLS codes without using any text data. You’ll just feed white space-delimited SNOMED codes and it will return the corresponding UMLS codes as a list. If there is no mapping, the original code is returned with no mapping.

      {'snomed': ['733187009', '449433008', '51264003'],
         'umls': ['C4546029', 'C3164619', 'C0271267']}

  *Related Notebook*: [Healthcare Code Mapping](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/11.1.Healthcare_Code_Mapping.ipynb)


### 3.0.2

We are very excited to announce that **Spark NLP for Healthcare 3.0.2** has been released! This release includes bug fixes and some compatibility improvements.

#### Highlights

* Dictionaries for Obfuscator were augmented with more than 10K names.
* Improved support for spark 2.3 and spark 2.4.
* Bug fixes in `DrugNormalizer`.

#### New Features
Provide confidence scores for all available tags in `MedicalNerModel`,

##### MedicalNerModel before 3.0.2
```
[[named_entity, 0, 9, B-PROBLEM, [word -> Pneumonia, confidence -> 0.9998], []]
```
##### Now in Spark NLP for Healthcare 3.0.2
```
[[named_entity, 0, 9, B-PROBLEM, [B-PROBLEM -> 0.9998, I-TREATMENT -> 0.0, I-PROBLEM -> 0.0, I-TEST -> 0.0, B-TREATMENT -> 1.0E-4, word -> Pneumonia, B-TEST -> 0.0], []]
```

### 3.0.1

We are very excited to announce that **Spark NLP for Healthcare 3.0.1** has been released! 

#### Highlights:

* Fixed problem in Assertion Status internal tokenization (reported in Spark-NLP #2470).
* Fixes in the internal implementation of DeIdentificationModel/Obfuscator.
* Being able to disable the use of regexes in the Deidentification process 
* Other minor bug fixes & general improvements.

#### DeIdentificationModel Annotator

##### New `seed` parameter.
Now we have the possibility of using a seed to guide the process of obfuscating entities and returning the same result across different executions. To make that possible a new method setSeed(seed:Int) was introduced.

**Example:**
Return obfuscated documents in a repeatable manner based on the same seed.
##### Scala
```scala
deIdentification = DeIdentification()
      .setInputCols(Array("ner_chunk", "token", "sentence"))
      .setOutputCol("dei")
      .setMode("obfuscate")
      .setObfuscateRefSource("faker")
      .setSeed(10)
      .setIgnoreRegex(true)
```
##### Python
```python
de_identification = DeIdentification() \
            .setInputCols(["ner_chunk", "token", "sentence"]) \
            .setOutputCol("dei") \
            .setMode("obfuscate") \
            .setObfuscateRefSource("faker") \
            .setSeed(10) \
            .setIgnoreRegex(True)

```

This seed controls how the obfuscated values are picked from a set of obfuscation candidates. Fixing the seed allows the process to be replicated.

**Example:**

Given the following input to the deidentification:
```
"David Hale was in Cocke County Baptist Hospital. David Hale"
```

If the annotator is set up with a seed of 10:
##### Scala
```
val deIdentification = new DeIdentification()
      .setInputCols(Array("ner_chunk", "token", "sentence"))
      .setOutputCol("dei")
      .setMode("obfuscate")
      .setObfuscateRefSource("faker")
      .setSeed(10)
      .setIgnoreRegex(true)
```
##### Python
```python
de_identification = DeIdentification() \
            .setInputCols(["ner_chunk", "token", "sentence"]) \
            .setOutputCol("dei") \
            .setMode("obfuscate") \
            .setObfuscateRefSource("faker") \
            .setSeed(10) \
            .setIgnoreRegex(True)

```

The result will be the following for any execution,

```
"Brendan Kitten was in New Megan.Brendan Kitten"
```
Now if we set up a seed of 32,
##### Scala

```scala
val deIdentification = new DeIdentification()
      .setInputCols(Array("ner_chunk", "token", "sentence"))
      .setOutputCol("dei")
      .setMode("obfuscate")
      .setObfuscateRefSource("faker")
      .setSeed(32)
      .setIgnoreRegex(true)
```
##### Python
```python
de_identification = DeIdentification() \
            .setInputCols(["ner_chunk", "token", "sentence"]) \
            .setOutputCol("dei") \
            .setMode("obfuscate") \
            .setObfuscateRefSource("faker") \
            .setSeed(10) \
            .setIgnoreRegex(True)
```

The result will be the following for any execution,

```
"Louise Pear was in Lake Edward.Louise Pear"
```

##### New `ignoreRegex` parameter.
You can now choose to completely disable the use of regexes in the deidentification process by setting the setIgnoreRegex param to True.
**Example:**
##### Scala

```scala
DeIdentificationModel.setIgnoreRegex(true)
```
##### Python
```python
DeIdentificationModel().setIgnoreRegex(True)
```

The default value for this param is `False` meaning that regexes will be used by default.

##### New supported entities for Deidentification & Obfuscation:

We added new entities to the default supported regexes:

* `SSN - Social security number.`
* `PASSPORT - Passport id.`
* `DLN - Department of Labor Number.`
* `NPI - National Provider Identifier.`
* `C_CARD - The id number for credits card.`
* `IBAN - International Bank Account Number.`
* `DEA - DEA Registration Number, which is an identifier assigned to a health care provider by the United States Drug Enforcement Administration.`

We also introduced new Obfuscator cases for these new entities.


### 3.0.0

We are very excited to announce that **Spark NLP for Healthcare 3.0.0** has been released! This has been one of the biggest releases we have ever done and we are so proud to share this with our customers.

#### Highlights:

Spark NLP for Healthcare 3.0.0 extends the support for Apache Spark 3.0.x and 3.1.x major releases on Scala 2.12 with both Hadoop 2.7. and 3.2. We now support all 4 major Apache Spark and PySpark releases of 2.3.x, 2.4.x, 3.0.x, and 3.1.x helping the customers to migrate from earlier Apache Spark versions to newer releases without being worried about Spark NLP support.

#### Highlights:
* Support for Apache Spark and PySpark 3.0.x on Scala 2.12
* Support for Apache Spark and PySpark 3.1.x on Scala 2.12
* Migrate to TensorFlow v2.3.1 with native support for Java to take advantage of many optimizations for CPU/GPU and new features/models introduced in TF v2.x
* A brand new `MedicalNerModel` annotator to train & load the licensed clinical NER models.
*  **Two times faster NER and Entity Resolution** due to new batch annotation technique.
* Welcoming 9x new Databricks runtimes to our Spark NLP family:
  * Databricks 7.3
  * Databricks 7.3 ML GPU
  * Databricks 7.4
  * Databricks 7.4 ML GPU
  * Databricks 7.5
  * Databricks 7.5 ML GPU
  * Databricks 7.6
  * Databricks 7.6 ML GPU
  * Databricks 8.0
  * Databricks 8.0 ML (there is no GPU in 8.0)
  * Databricks 8.1 Beta
* Welcoming 2x new EMR 6.x series to our Spark NLP family:
  * EMR 6.1.0 (Apache Spark 3.0.0 / Hadoop 3.2.1)
  * EMR 6.2.0 (Apache Spark 3.0.1 / Hadoop 3.2.1)
* Starting Spark NLP for Healthcare 3.0.0 the default packages  for CPU and GPU will be based on Apache Spark 3.x and Scala 2.12.

##### Deprecated

Text2SQL annotator is deprecated and will not be maintained going forward. We are working on a better and faster version of Text2SQL at the moment and will announce soon.

#### 1. MedicalNerModel Annotator

Starting Spark NLP for Healthcare 3.0.0, the licensed clinical and biomedical pretrained NER models will only work with this brand new annotator called `MedicalNerModel` and will not work with `NerDLModel` in open source version.

In order to make this happen, we retrained all the clinical NER models (more than 80) and uploaded to models hub.

Example:


    clinical_ner = MedicalNerModel.pretrained("ner_clinical", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"])\
    .setOutputCol("ner")

#### 2. Speed Improvements

A new batch annotation technique implemented in Spark NLP 3.0.0 for `NerDLModel`,`BertEmbeddings`, and `BertSentenceEmbeddings` annotators will be reflected in `MedicalNerModel` and it improves prediction/inferencing performance radically. From now on the `batchSize` for these annotators means the number of rows that can be fed into the models for prediction instead of sentences per row. You can control the throughput when you are on accelerated hardware such as GPU to fully utilise it. Here are the overall speed comparison:

Now, NER inference and Entity Resolution are **two times faster** on CPU and three times faster on GPU.


#### 3. JSL Clinical NER Model

We are releasing the richest clinical NER model ever, spanning over 80 entities. It has been under development for the last 6 months and we manually annotated more than 4000 clinical notes to cover such a high number of entities in a single model. It has 4 variants at the moment:

- `jsl_ner_wip_clinical`
- `jsl_ner_wip_greedy_clinical`
- `jsl_ner_wip_modifier_clinical`
- `jsl_rd_ner_wip_greedy_clinical`

##### Entities:

`Kidney_Disease`, `HDL`, `Diet`, `Test`, `Imaging_Technique`, `Triglycerides`, `Obesity`, `Duration`, `Weight`, `Social_History_Header`, `ImagingTest`, `Labour_Delivery`, `Disease_Syndrome_Disorder`, `Communicable_Disease`, `Overweight`, `Units`, `Smoking`, `Score`, `Substance_Quantity`, `Form`, `Race_Ethnicity`, `Modifier`, `Hyperlipidemia`, `ImagingFindings`, `Psychological_Condition`, `OtherFindings`, `Cerebrovascular_Disease`, `Date`, `Test_Result`, `VS_Finding`, `Employment`, `Death_Entity`, `Gender`, `Oncological`, `Heart_Disease`, `Medical_Device`, `Total_Cholesterol`, `ManualFix`, `Time`, `Route`, `Pulse`, `Admission_Discharge`, `RelativeDate`, `O2_Saturation`, `Frequency`, `RelativeTime`, `Hypertension`, `Alcohol`, `Allergen`, `Fetus_NewBorn`, `Birth_Entity`, `Age`, `Respiration`, `Medical_History_Header`, `Oxygen_Therapy`, `Section_Header`, `LDL`, `Treatment`, `Vital_Signs_Header`, `Direction`, `BMI`, `Pregnancy`, `Sexually_Active_or_Sexual_Orientation`, `Symptom`, `Clinical_Dept`, `Measurements`, `Height`, `Family_History_Header`, `Substance`, `Strength`, `Injury_or_Poisoning`, `Relationship_Status`, `Blood_Pressure`, `Drug`, `Temperature`, `EKG_Findings`, `Diabetes`, `BodyPart`, `Vaccine`, `Procedure`, `Dosage`


#### 4. JSL Clinical Assertion Model

We are releasing a brand new clinical assertion model, supporting 8 assertion statuses.

- `jsl_assertion_wip`


##### Assertion Labels :

`Present`, `Absent`, `Possible`, `Planned`, `Someoneelse`, `Past`, `Family`, `Hypotetical`

#### 5. Library Version Compatibility Table :

Spark NLP for Healthcare 3.0.0 is compatible with Spark NLP 3.0.1

#### 6. Pretrained Models Version Control (Beta):

Due to active release cycle, we are adding & training new pretrained models at each release and it might be tricky to maintain the backward compatibility or keep up with the latest models, especially for the users using our models locally in air-gapped networks.

We are releasing a new utility class to help you check your local & existing models with the latest version of everything we have up to date. You will not need to specify your AWS credentials from now on. This is the second version of the model checker we released with 2.7.6 and will replace that soon.

    from sparknlp_jsl.compatibility_beta import CompatibilityBeta

    compatibility = CompatibilityBeta(spark)

    print(compatibility.findVersion("ner_deid"))


#### 7. Updated Pretrained Models:

 (requires fresh `.pretraned()`)

None


### 2.7.6

We are glad to announce that Spark NLP for Healthcare 2.7.6 has been released!

#### Highlights:

- New pretrained **Radiology Assertion Status** model to assign `Confirmed`, `Suspected`, `Negative` assertion scopes to imaging findings or any clinical tests.

- **Obfuscating** the same sensitive information (patient or doctor name) with the same fake names across the same clinical note.

- Version compatibility checker for the pretrained clinical models and builds to keep up with the latest development efforts in production.

- Adding more English names to faker module in **Deidentification**.

- Updated & improved clinical **SentenceDetectorDL** model.

- New upgrades on `ner_deid_large` and `ner_deid_enriched` NER models to cover more use cases with better resolutions.

- Adding more [examples](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/scala/healthcare) to workshop repo for _Scala_ users to practice more on healthcare annotators. 

- Bug fixes & general improvements.

#### 1. Radiology Assertion Status Model

We trained a new assertion model to assign `Confirmed`, `Suspected`, `Negative` assertion scopes to imaging findings or any clinical tests. It will try to assign these statuses to any named entity you would feed to the assertion annotater in the same pipeline.

     radiology_assertion = AssertionDLModel.pretrained("assertion_dl_radiology", "en", "clinical/models")\
     .setInputCols(["sentence", "ner_chunk", "embeddings"])\
     .setOutputCol("assertion")

`text = Blunting of the left costophrenic angle on the lateral view posteriorly suggests a small left pleural effusion. No right-sided pleural effusion or pneumothorax is definitively seen. There are mildly displaced fractures of the left lateral 8th and likely 9th ribs.`


|sentences |chunk | ner_label |sent_id|assertion|
|-----------:|:-----:|:---------:|:------:|:------|
|Blunting of the left costophrenic angle on the lateral view posteriorly suggests a small left pleural effusion.|Blunting           |ImagingFindings|0      |Confirmed|
|Blunting of the left costophrenic angle on the lateral view posteriorly suggests a small left pleural effusion.|effusion           |ImagingFindings|0      |Suspected|
|No right-sided pleural effusion or pneumothorax is definitively seen.                                          |effusion           |ImagingFindings|1      |Negative |
|No right-sided pleural effusion or pneumothorax is definitively seen.                                          |pneumothorax       |ImagingFindings|1      |Negative |
|There are mildly displaced fractures of the left lateral 8th and likely 9th ribs.                              |displaced fractures|ImagingFindings|2      |Confirmed|

You can also use this with `AssertionFilterer` to return clinical findings from a note only when it is i.e. `confirmed` or `suspected`.

     assertion_filterer = AssertionFilterer()\
     .setInputCols("sentence","ner_chunk","assertion")\
     .setOutputCol("assertion_filtered")\
     .setWhiteList(["confirmed","suspected"])

     >> ["displaced fractures", "effusion"]

#### 2. **Obfuscating** with the same fake name across the same note:

     obfuscation = DeIdentification()\
      .setInputCols(["sentence", "token", "ner_chunk"]) \
      .setOutputCol("deidentified") \
      .setMode("obfuscate")\
      .setObfuscateDate(True)\
      .setSameEntityThreshold(0.8)\
      .setObfuscateRefSource("faker")

      
    text =''' Provider: David Hale, M.D.
              Pt: Jessica Parker 
              David told  Jessica that she will need to visit the clinic next month.'''



|    | sentence                                                               | obfuscated                                                          |
|---:|:-----------------------------------------------------------------------|:----------------------------------------------------------------------|
|  0 | Provider: `David Hale`, M.D.                                             | Provider: `Dennis Perez`, M.D.                                          |
|  1 | Pt: `Jessica Parker`                                                     | Pt: `Gerth Bayer`                                                       |
|  2 | `David` told  `Jessica` that she will need to visit the clinic next month. | `Dennis` told  `Gerth` that she will need to visit the clinic next month. |

#### 3. Library Version Compatibility Table :

We are releasing the version compatibility table to help users get to see which Spark NLP licensed version is built against which core (open source) version. We are going to release a detailed one after running some tests across the jars from each library.

| Healthcare| Public |
|-----------|--------|
| 2.7.6     | 2.7.4  |
| 2.7.5     | 2.7.4  |
| 2.7.4     | 2.7.3  |
| 2.7.3     | 2.7.3  |
| 2.7.2     | 2.6.5  |
| 2.7.1     | 2.6.4  |
| 2.7.0     | 2.6.3  |
| 2.6.2     | 2.6.2  |
| 2.6.0     | 2.6.0  |
| 2.5.5     | 2.5.5  |
| 2.5.3     | 2.5.3  |
| 2.5.2     | 2.5.2  |
| 2.5.0     | 2.5.0  |
| 2.4.7     | 2.4.5  |
| 2.4.6     | 2.4.5  |
| 2.4.5     | 2.4.5  |
| 2.4.2     | 2.4.2  |
| 2.4.1     | 2.4.1  |
| 2.4.0     | 2.4.0  |
| 2.3.6     | 2.3.6  |
| 2.3.5     | 2.3.5  |
| 2.3.4     | 2.3.4  |


#### 4. Pretrained Models Version Control :

Due to active release cycle, we are adding & training new pretrained models at each release and it might be tricky to maintain the backward compatibility or keep up with the latest models, especially for the users using our models locally in air-gapped networks. 

We are releasing a new utility class to help you check your local & existing models with the latest version of everything we have up to date. This is an highly experimental feature of which we plan to improve and add more capability later on.


    from sparknlp_jsl.check_compatibility import Compatibility

     checker = sparknlp_jsl.Compatibility()

     result = checker.find_version(aws_access_key_id=license_keys['AWS_ACCESS_KEY_ID'],
                            aws_secret_access_key=license_keys['AWS_SECRET_ACCESS_KEY'],
                            metadata_path=None,
                            model = 'all' , # or a specific model name
                            target_version='all',
                            cache_pretrained_path='/home/ubuntu/cache_pretrained')

     >> result['outdated_models']
     
      [{'model_name': 'clinical_ner_assertion',
        'current_version': '2.4.0',
        'latest_version': '2.6.4'},
       {'model_name': 'jsl_rd_ner_wip_greedy_clinical',
        'current_version': '2.6.1',
        'latest_version': '2.6.2'},
       {'model_name': 'ner_anatomy',
        'current_version': '2.4.2',
        'latest_version': '2.6.4'},
       {'model_name': 'ner_aspect_based_sentiment',
        'current_version': '2.6.2',
        'latest_version': '2.7.2'},
       {'model_name': 'ner_bionlp',
        'current_version': '2.4.0',
        'latest_version': '2.7.0'},
       {'model_name': 'ner_cellular',
        'current_version': '2.4.2',
        'latest_version': '2.5.0'}]

      >> result['version_comparison_dict']
      
      [{'clinical_ner_assertion': {'current_version': '2.4.0', 'latest_version': '2.6.4'}}, {'jsl_ner_wip_clinical': {'current_version': '2.6.5', 'latest_version': '2.6.1'}}, {'jsl_ner_wip_greedy_clinical': {'current_version': '2.6.5', 'latest_version': '2.6.5'}}, {'jsl_ner_wip_modifier_clinical': {'current_version': '2.6.4', 'latest_version': '2.6.4'}}, {'jsl_rd_ner_wip_greedy_clinical': {'current_version': '2.6.1','latest_version': '2.6.2'}}]

#### 5. Updated Pretrained Models:

 (requires fresh `.pretraned()`)
 
- ner_deid_large
- ner_deid_enriched


### 2.7.5

We are glad to announce that Spark NLP for Healthcare 2.7.5 has been released!  

#### Highlights:

- New pretrained **Relation Extraction** model to link clinical tests to test results and dates to clinical entities: `re_test_result_date`
- Adding two new `Admission` and `Discharge` entities to `ner_events_clinical` and renaming it to `ner_events_admission_clinical`
- Improving `ner_deid_enriched` NER model to cover `Doctor` and `Patient` name entities in various context and notations.
- Bug fixes & general improvements.

#### 1. re_test_result_date :

text = "Hospitalized with pneumonia in June, confirmed by a positive PCR of any specimen, evidenced by SPO2 </= 93% or PaO2/FiO2 < 300 mmHg"

|    | Chunk-1   | Entity-1   | Chunk-2   | Entity-2    | Relation     |
|---:|:----------|:-----------|:----------|:------------|:-------------|
|  0 | pneumonia | Problem    | june      | Date        | is_date_of   |
|  1 | PCR       | Test       | positive  | Test_Result | is_result_of |
|  2 | SPO2      | Test       | 93%       | Test_Result | is_result_of |
|  3 | PaO2/FiO2 | Test       | 300 mmHg  | Test_Result | is_result_of |

#### 2. `ner_events_admission_clinical` :

`ner_events_clinical` NER model is updated & improved to include `Admission` and `Discharge` entities.

text ="She is diagnosed as cancer in 1991. Then she was admitted to Mayo Clinic in May 2000 and discharged in October 2001"

|    | chunk        | entity        |
|---:|:-------------|:--------------|
|  0 | diagnosed    | OCCURRENCE    |
|  1 | cancer       | PROBLEM       |
|  2 | 1991         | DATE          |
|  3 | admitted     | ADMISSION     |
|  4 | Mayo Clinic  | CLINICAL_DEPT |
|  5 | May 2000     | DATE          |
|  6 | discharged   | DISCHARGE     |
|  7 | October 2001 | DATE          |


#### 3. Improved `ner_deid_enriched` :

PHI NER model is retrained to cover `Doctor` and `Patient` name entities even there is a punctuation between tokens as well as all upper case or lowercased.

text ="A . Record date : 2093-01-13 , DAVID HALE , M.D . , Name : Hendrickson , Ora MR . # 7194334 Date : 01/13/93 PCP : Oliveira , 25 month years-old , Record date : 2079-11-09 . Cocke County Baptist Hospital . 0295 Keats Street"

|    | chunk                         | entity        |
|---:|:------------------------------|:--------------|
|  0 | 2093-01-13                    | MEDICALRECORD |
|  1 | DAVID HALE                    | DOCTOR        |
|  2 | Hendrickson , Ora             | PATIENT       |
|  3 | 7194334                       | MEDICALRECORD |
|  4 | 01/13/93                      | DATE          |
|  5 | Oliveira                      | DOCTOR        |
|  6 | 25                            | AGE           |
|  7 | 2079-11-09                    | MEDICALRECORD |
|  8 | Cocke County Baptist Hospital | HOSPITAL      |
|  9 | 0295 Keats Street             | STREET        |


### 2.7.4

We are glad to announce that Spark NLP for Healthcare 2.7.4 has been released!  

#### Highlights:

- Introducing a new annotator to extract chunks with NER tags using regex-like patterns: **NerChunker**.
- Introducing two new annotators to filter chunks: **ChunkFilterer** and **AssertionFilterer**.
- Ability to change the entity type in **NerConverterInternal** without using ChunkMerger (`setReplaceDict`).
- In **DeIdentification** model, ability to use `faker` and static look-up lists at the same time randomly in `Obfuscation` mode.
- New **De-Identification NER** model, augmented with synthetic datasets to detect uppercased name entities.
- Bug fixes & general improvements.

#### 1. NerChunker:

Similar to what we used to do in **POSChunker** with POS tags, now we can also extract phrases that fits into a known pattern using the NER tags. **NerChunker** would be quite handy to extract entity groups with neighboring tokens when there is no pretrained NER model to address certain issues. Lets say we want to extract clinical findings and body parts together as a single chunk even if there are some unwanted tokens between. 

**How to use:**

    ner_model = NerDLModel.pretrained("ner_radiology", "en", "clinical/models")\
        .setInputCols("sentence","token","embeddings")\
        .setOutputCol("ner")
                
    ner_chunker = NerChunker().\
        .setInputCols(["sentence","ner"])\
        .setOutputCol("ner_chunk")\
        .setRegexParsers(["<IMAGINGFINDINGS>*<BODYPART>"])

    text = 'She has cystic cyst on her kidney.'

    >> ner tags: [(cystic, B-IMAGINGFINDINGS), (cyst,I-IMAGINGFINDINGS), (kidney, B-BODYPART)
    >> ner_chunk: ['cystic cyst on her kidney']


#### 2. ChunkFilterer:

**ChunkFilterer** will allow you to filter out named entities by some conditions or predefined look-up lists, so that you can feed these entities to other annotators like Assertion Status or Entity Resolvers.  It can be used with two criteria: **isin** and **regex**.

**How to use:**

    ner_model = NerDLModel.pretrained("ner_clinical", "en", "clinical/models")\
          .setInputCols("sentence","token","embeddings")\
          .setOutputCol("ner")

    ner_converter = NerConverter() \
          .setInputCols(["sentence", "token", "ner"]) \
          .setOutputCol("ner_chunk")
          
    chunk_filterer = ChunkFilterer()\
          .setInputCols("sentence","ner_chunk")\
          .setOutputCol("chunk_filtered")\
          .setCriteria("isin") \ 
          .setWhiteList(['severe fever','sore throat'])

    text = 'Patient with severe fever, sore throat, stomach pain, and a headache.'

    >> ner_chunk: ['severe fever','sore throat','stomach pain','headache']
    >> chunk_filtered: ['severe fever','sore throat']


#### 3. AssertionFilterer:

**AssertionFilterer** will allow you to filter out the named entities by the list of acceptable assertion statuses. This annotator would be quite handy if you want to set a white list for the acceptable assertion statuses like `present` or `conditional`; and do not want `absent` conditions get out of your pipeline.

**How to use:**

    clinical_assertion = AssertionDLModel.pretrained("assertion_dl", "en", "clinical/models") \
      .setInputCols(["sentence", "ner_chunk", "embeddings"]) \
      .setOutputCol("assertion")
    
    assertion_filterer = AssertionFilterer()\
      .setInputCols("sentence","ner_chunk","assertion")\
      .setOutputCol("assertion_filtered")\
      .setWhiteList(["present"])


    text = 'Patient with severe fever and sore throat, but no stomach pain.'

    >> ner_chunk: ['severe fever','sore throat','stomach pain','headache']
    >> assertion_filtered: ['severe fever','sore throat']

 
### 2.7.3

We are glad to announce that Spark NLP for Healthcare 2.7.3 has been released!  

#### Highlights:

- Introducing a brand-new **RelationExtractionDL Annotator** – Achieving SOTA results in clinical relation extraction using **BioBert**.
- Massive Improvements & feature enhancements in **De-Identification** module:
    - Introduction of **faker** augmentation in Spark NLP for Healthcare to generate random data for obfuscation in de-identification module.
    - Brand-new annotator for **Structured De-Identification**.
- **Drug Normalizer:**  Normalize medication-related phrases (dosage, form and strength) and abbreviations in text and named entities extracted by NER models.
- **Confidence scores** in **assertion** output : just like NER output, assertion models now also support confidence scores for each prediction.
- **Cosine similarity** metrics in entity resolvers to get more informative and semantically correct results.
- **AuxLabel** in the metadata of entity resolvers to return additional mappings.
- New **Relation Extraction** models to extract relations between **body parts** and clinical entities.
- New **Entity Resolver** models to extract billable medical codes.
- New **Clinical Pretrained NER** models.
- Bug fixes & general improvements.
- Matching the version with Spark NLP open-source v2.7.3.


#### 1. Improvements in De-Identification Module:

Integration of `faker` library to automatically generate random data like names, dates, addresses etc so users dont have to specify dummy data (custom obfuscation files can still be used). It also improves the obfuscation results due to a bigger pool of random values.

**How to use:**

Set the flag `setObfuscateRefSource` to `faker`

    deidentification = DeIdentification()
        .setInputCols(["sentence", "token", "ner_chunk"])\
		.setOutputCol("deidentified")\
		.setMode("obfuscate") \
		.setObfuscateRefSource("faker")

For more details: Check out this [notebook ](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/4.Clinical_DeIdentification.ipynb)

#### 2. Structured De-Identification Module:

Introduction of a new annotator to handle de-identification of structured data. it  allows users to define a mapping of columns and their obfuscation policy. Users can also provide dummy data and map them to columns they want to replace values in.

**How to use:**

	obfuscator = StructuredDeidentification \
		(spark,{"NAME":"PATIENT","AGE":"AGE"},
		obfuscateRefSource = "faker")

	obfuscator_df = obfuscator.obfuscateColumns(df)

	obfuscator_df.select("NAME","AGE").show(truncate=False)

**Example:**

Input Data:

Name  | Age
------------- | -------------
Cecilia Chapman|83
Iris Watson    |9  
Bryar Pitts    |98
Theodore Lowe  |16
Calista Wise   |76

Deidentified:

Name  | Age
------------- | -------------
Menne Erdôs|20
 Longin Robinson     |31  
 Flynn Fiedlerová   |50
 John Wakeland  |21
Vanessa Andersson   |12


For more details: Check out this [notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/4.Clinical_DeIdentification.ipynb).

#### 3. Introducing SOTA relation extraction model using BioBert

A brand-new end-to-end trained BERT model, resulting in massive improvements. Another new annotator (`ReChunkFilter`) is also developed for this new model to allow syntactic features work well with BioBert to extract relations.

**How to use:**

    re_ner_chunk_filter = RENerChunksFilter()\
        .setInputCols(["ner_chunks", "dependencies"])\
        .setOutputCol("re_ner_chunks")\
        .setRelationPairs(pairs)\
        .setMaxSyntacticDistance(4)

    re_model = RelationExtractionDLModel()\
        .pretrained(“redl_temporal_events_biobert”, "en", "clinical/models")\
        .setPredictionThreshold(0.9)\
        .setInputCols(["re_ner_chunks", "sentences"])\
        .setOutputCol("relations")


##### Benchmarks:

**on benchmark datasets**

model                           | Spark NLP ML model | Spark NLP DL model | benchmark
---------------------------------|-----------|-----------|-----------
re_temporal_events_clinical     | 68.29     | 71.0      | 80.2 [1](https://arxiv.org/pdf/2012.08790.pdf)
re_clinical                     | 56.45     | **69.2**      | 68.2      [2](ncbi.nlm.nih.gov/pmc/articles/PMC7153059/)
re_human_pheotype_gene_clinical | -         | **87.9**      | 67.2 [3](https://arxiv.org/pdf/1903.10728.pdf)
re_drug_drug_interaction        | -         | 72.1      | 83.8 [4](https://www.aclweb.org/anthology/2020.knlp-1.4.pdf)
re_chemprot                     | 76.69     | **94.1**      | 83.64 [5](https://www.aclweb.org/anthology/D19-1371.pdf)

**on in-house annotations**

model                           | Spark NLP ML model | Spark NLP DL model
---------------------------------|-----------|-----------
re_bodypart_problem             | 84.58     | 85.7
re_bodypart_procedure           | 61.0      | 63.3
re_date_clinical                | 83.0      | 84.0  
re_bodypart_direction           | 93.5      | 92.5  


For more details: Check out the [notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/10.1.Clinical_Relation_Extraction_BodyParts_Models.ipynb) or [modelshub](https://nlp.johnsnowlabs.com/models?tag=relation_extraction).



#### 4. Drug Normalizer:

Standardize units of drugs and handle abbreviations in raw text or drug chunks identified by any NER model. This normalization significantly improves performance of entity resolvers.  

**How to use:**

    drug_normalizer = DrugNormalizer()\
        .setInputCols("document")\
        .setOutputCol("document_normalized")\
        .setPolicy("all") #all/abbreviations/dosages

**Examples:**

`drug_normalizer.transform("adalimumab 54.5 + 43.2 gm”)`

    >>> "adalimumab 97700 mg"

**Changes:** _combine_ `54.5` + `43.2` and _normalize_ `gm` to `mg`

`drug_normalizer.transform("Agnogenic one half cup”)`

    >>> "Agnogenic 0.5 oral solution"

**Changes:** _replace_ `one half` to the `0.5`, _normalize_ `cup` to the `oral solution`

`drug_normalizer.transform("interferon alfa-2b 10 million unit ( 1 ml ) injec”)`

    >>> "interferon alfa - 2b 10000000 unt ( 1 ml ) injection "

**Changes:** _convert_ `10 million unit` to the `10000000 unt`, _replace_ `injec` with `injection`

For more details: Check out this [notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/23.Drug_Normalizer.ipynb)

#### 5. Assertion models to support confidence in output:

Just like NER output, assertion models now also provides _confidence scores_ for each prediction.

chunks  | entities | assertion | confidence
-----------|-----------|------------|-------------
a headache | PROBLEM | present |0.9992
anxious | PROBLEM | conditional |0.9039
alopecia | PROBLEM | absent |0.9992
pain | PROBLEM | absent |0.9238

`.setClasses()` method is deprecated in `AssertionDLApproach`  and users do not need to specify number of classes while training, as it will be inferred from the dataset.


#### 6. New Relation Extraction Models:


We are also releasing new relation extraction models to link the clinical entities to body parts and dates. These models are trained using binary relation extraction approach for better accuracy.

**- re_bodypart_direction :**  Relation Extraction between `Body Part` and `Direction` entities.

**Example:**

**Text:** _“MRI demonstrated infarction in the upper brain stem , left cerebellum and right basil ganglia”_

relations | entity1                     | chunk1     | entity2                     | chunk2        | confidence
-----------|-----------------------------|------------|-----------------------------|---------------|------------
1         | Direction                   | upper      | bodyPart | brain stem    | 0.999
0         | Direction                   | upper      | bodyPart | cerebellum    | 0.999
0         | Direction                   | upper      | bodyPart | basil ganglia | 0.999
0         | bodyPart | brain stem | Direction                   | left          | 0.999
0         | bodyPart | brain stem | Direction                   | right         | 0.999
1         | Direction                   | left       | bodyPart | cerebellum    | 1.0         
0         | Direction                   | left       | bodyPart | basil ganglia | 0.976
0         | bodyPart | cerebellum | Direction                   | right         | 0.953
1         | Direction                   | right      | bodyPart | basil ganglia | 1.0       


**- re_bodypart_problem :** Relation Extraction between `Body Part` and `Problem` entities.

**Example:**

**Text:** _“No neurologic deficits other than some numbness in his left hand.”_

relation | entity1   | chunk1              | entity2                      | chunk2   | confidence  
--|------------------|--------------------|-----------------------------|---------|-----------  
0 | Symptom   | neurologic deficits | bodyPart | hand     |          1
1 | Symptom   | numbness            | bodyPart | hand     |          1


**- re_bodypart_proceduretest :**  Relation Extraction between `Body Part` and `Procedure`, `Test` entities.

**Example:**

**Text:** _“TECHNIQUE IN DETAIL: After informed consent was obtained from the patient and his mother, the chest was scanned with portable ultrasound.”_

relation | entity1                      | chunk1   | entity2   | chunk2              | confidence  
---------|-----------------------------|---------|----------|--------------------|-----------
1 | bodyPart | chest    | Test      | portable ultrasound |    0.999

**-re_date_clinical :** Relation Extraction between `Date` and different clinical entities.   

**Example:**

**Text:** _“This 73 y/o patient had CT on 1/12/95, with progressive memory and cognitive decline since 8/11/94.”_

relations | entity1 | chunk1                                   | entity2 | chunk2  | confidence
-----------|---------|------------------------------------------|---------|---------|------------|
1         | Test    | CT                                       | Date    | 1/12/95 | 1.0
1         | Symptom | progressive memory and cognitive decline | Date    | 8/11/94 | 1.0


**How to use:**

    re_model = RelationExtractionModel()\
        .pretrained("re_bodypart_direction","en","clinical/models")\
        .setInputCols(["embeddings", "pos_tags", "ner_chunks", "dependencies"])\
        .setOutputCol("relations")\
        .setMaxSyntacticDistance(4)\
        .setRelationPairs([‘Internal_organ_or_component’, ‘Direction’])



For more details: Check out the [notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/10.1.Clinical_Relation_Extraction_BodyParts_Models.ipynb) or [modelshub](https://nlp.johnsnowlabs.com/models?tag=relation_extraction).


**New matching scheme for entity resolvers - improved accuracy:** Adding the option to use `cosine similarity` to resolve entities and find closest matches, resulting in better, more semantically correct results.

#### 7. New Resolver Models using `JSL SBERT`:

+ `sbiobertresolve_icd10cm_augmented`

+ `sbiobertresolve_cpt_augmented`

+ `sbiobertresolve_cpt_procedures_augmented`

+ `sbiobertresolve_icd10cm_augmented_billable_hcc`

+ `sbiobertresolve_hcc_augmented`


**Returning auxilary columns mapped to resolutions:**  Chunk entity resolver and sentence entity resolver now returns auxilary data that is mapped the resolutions during training. This will allow users to get multiple resolutions with single model without using any other annotator in the pipeline (In order to get billable codes otherwise there needs to be other modules in the same pipeline)

**Example:**

`sbiobertresolve_icd10cm_augmented_billable_hcc`
**Input Text:** _"bladder cancer"_

idx | chunks | code | resolutions | all_codes | billable | hcc_status | hcc_score | all_distances   
---|---------------|-------|-------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------|--------------------------|--------------------------|---------------------------|---------------------------------------------------
 0 | bladder cancer | C679 | ['bladder cancer', 'suspected bladder cancer', 'cancer in situ of urinary bladder', 'tumor of bladder neck', 'malignant tumour of bladder neck'] | ['C679', 'Z126', 'D090', 'D494', 'C7911'] | ['1', '1', '1', '1', '1'] | ['1', '0', '0', '0', '1'] | ['11', '0', '0', '0', '8'] | ['0.0000', '0.0904', '0.0978', '0.1080', '0.1281'] |

`sbiobertresolve_cpt_augmented`  
**Input Text:** _"ct abdomen without contrast"_

idx|  cpt code |   distance |resolutions                                                       
---:|------:|-----------:|:-------------------------------------------------------------------
0 | 74150 |     0.0802 | Computed tomography, abdomen; without contrast material            |
1 | 65091 |     0.1312 | Evisceration of ocular contents; without implant                   |
2 | 70450 |     0.1323 | Computed tomography, head or brain; without contrast material      |
3 | 74176 |     0.1333 | Computed tomography, abdomen and pelvis; without contrast material |
4 | 74185 |     0.1343 | Magnetic resonance imaging without contrast                        |
5 | 77059 |     0.1343 | Magnetic resonance imaging without contrast                        |

#### 8. New Pretrained Clinical NER Models

+ NER Radiology
**Input Text:** _"Bilateral breast ultrasound was subsequently performed, which demonstrated an ovoid mass measuring approximately 0.5 x 0.5 x 0.4 cm in diameter located within the anteromedial aspect of the left shoulder. This mass demonstrates isoechoic echotexture to the adjacent muscle, with no evidence of internal color flow. This may represent benign fibrous tissue or a lipoma."_

|idx | chunks                | entities                  |
|----|-----------------------|---------------------------|
| 0  | Bilateral             | Direction                 |
| 1  | breast                | BodyPart                  |
| 2  | ultrasound            | ImagingTest               |
| 3  | ovoid mass            | ImagingFindings           |
| 4  | 0.5 x 0.5 x 0.4       | Measurements              |
| 5  | cm                    | Units                     |
| 6  | anteromedial aspect   | Direction                 |
| 7  | left                  | Direction                 |
| 8  | shoulder              | BodyPart                  |
| 9  | mass                  | ImagingFindings           |
| 10 | isoechoic echotexture | ImagingFindings           |
| 11 | muscle                | BodyPart                  |
| 12 | internal color flow   | ImagingFindings           |
| 13 | benign fibrous tissue | ImagingFindings           |
| 14 | lipoma                | Disease_Syndrome_Disorder |



### 2.7.2

We are glad to announce that Spark NLP for Healthcare 2.7.2 has been released !

In this release, we introduce the following features:

+ Far better accuracy for resolving medication terms to RxNorm codes:

  `ondansetron 8 mg tablet' -> '312086`
+ Far better accuracy for resolving diagnosis terms to ICD-10-CM codes:

 `TIA -> transient ischemic attack (disorder)	‘S0690’`
+ New ability to map medications to pharmacological actions (PA):

  `'metformin' -> ‘Hypoglycemic Agents’ `
+ 2 new *greedy* named entity recognition models for medication details:

 `ner_drugs_greedy: ‘magnesium hydroxide 100mg/1ml PO’`

 ` ner_posology _greedy: ‘12 units of insulin lispro’ `

+ New model to *classify the gender* of a patient in a given medical note:

 `'58yo patient with a family history of breast cancer' -> ‘female’ `
+ And starting customized spark sessions with rich parameters

```python
        params = {"spark.driver.memory":"32G",
        "spark.kryoserializer.buffer.max":"2000M",
        "spark.driver.maxResultSize":"2000M"}

        spark = sparknlp_jsl.start(secret, params=params)
```
State-of-the-art accuracy is achieved using new healthcare-tuned BERT Sentence Embeddings (s-Bert). The following sections include more details, metrics, and examples.

#### Named Entity Recognizers for Medications


+ A new medication NER (`ner_drugs_greedy`) that joins the drug entities with neighboring entities such as  `dosage`, `route`, `form` and `strength`; and returns a single entity `drug`.  This greedy NER model would be highly useful if you want to extract a drug with its context and then use it to get a RxNorm code (drugs may get different RxNorm codes based on the dosage and strength information).

###### Metrics

| label  | tp    | fp   | fn   | prec  | rec   | f1    |
|--------|-------|------|------|-------|-------|-------|
| I-DRUG | 37423 | 4179 | 3773 | 0.899 | 0.908 | 0.904 |
| B-DRUG | 29699 | 2090 | 1983 | 0.934 | 0.937 | 0.936 |


+ A new medication NER (`ner_posology_greedy`) that joins the drug entities with neighboring entities such as  `dosage`, `route`, `form` and `strength`.  It also returns all the other medication entities even if not related to (or joined with) a drug.   

Now we have five different medication-related NER models. You can see the outputs from each model below:

Text = ''*The patient was prescribed 1 capsule of Advil 10 mg for 5 days and magnesium hydroxide 100mg/1ml suspension PO. He was seen by the endocrinology service and she was discharged on 40 units of insulin glargine at night, 12 units of insulin lispro with meals, and metformin 1000 mg two times a day.*''

a. **ner_drugs_greedy**

|   | chunks                           | begin | end | entities |
| - | -------------------------------- | ----- | --- | -------- |
| 0 | 1 capsule of Advil 10 mg         | 27    | 50  | DRUG     |
| 1 | magnesium hydroxide 100mg/1ml PO | 67    | 98  | DRUG     |
| 2 |  40 units of insulin glargine    | 168   | 195 | DRUG     |
| 3 |  12 units of insulin lispro      | 207   | 232 | DRUG     |

b. **ner_posology_greedy**

|    | chunks                           |   begin |   end | entities   |
|---:|:---------------------------------|--------:|------:|:-----------|
|  0 | 1 capsule of Advil 10 mg         |      27 |    50 | DRUG       |
|  1 | magnesium hydroxide 100mg/1ml PO |      67 |    98 | DRUG       |
|  2 | for 5 days                       |      52 |    61 | DURATION   |
|  3 | 40 units of insulin glargine     |     168 |   195 | DRUG       |
|  4 | at night                         |     197 |   204 | FREQUENCY  |
|  5 | 12 units of insulin lispro       |     207 |   232 | DRUG       |
|  6 | with meals                       |     234 |   243 | FREQUENCY  |
|  7 | metformin 1000 mg                |     250 |   266 | DRUG       |
|  8 | two times a day                  |     268 |   282 | FREQUENCY  |

c. **ner_drugs**

|    | chunks              |   begin |   end | entities   |
|---:|:--------------------|--------:|------:|:-----------|
|  0 | Advil               |      40 |    44 | DrugChem   |
|  1 | magnesium hydroxide |      67 |    85 | DrugChem   |
|  2 | metformin           |     261 |   269 | DrugChem   |


d.**ner_posology**

|    | chunks              |   begin |   end | entities   |
|---:|:--------------------|--------:|------:|:-----------|
|  0 | 1                   |      27 |    27 | DOSAGE     |
|  1 | capsule             |      29 |    35 | FORM       |
|  2 | Advil               |      40 |    44 | DRUG       |
|  3 | 10 mg               |      46 |    50 | STRENGTH   |
|  4 | for 5 days          |      52 |    61 | DURATION   |
|  5 | magnesium hydroxide |      67 |    85 | DRUG       |
|  6 | 100mg/1ml           |      87 |    95 | STRENGTH   |
|  7 | PO                  |      97 |    98 | ROUTE      |
|  8 | 40 units            |     168 |   175 | DOSAGE     |
|  9 | insulin glargine    |     180 |   195 | DRUG       |
| 10 | at night            |     197 |   204 | FREQUENCY  |
| 11 | 12 units            |     207 |   214 | DOSAGE     |
| 12 | insulin lispro      |     219 |   232 | DRUG       |
| 13 | with meals          |     234 |   243 | FREQUENCY  |
| 14 | metformin           |     250 |   258 | DRUG       |
| 15 | 1000 mg             |     260 |   266 | STRENGTH   |
| 16 | two times a day     |     268 |   282 | FREQUENCY  |

e. **ner_drugs_large**

|    | chunks                            |   begin |   end | entities   |
|---:|:----------------------------------|--------:|------:|:-----------|
|  0 | Advil 10 mg                       |      40 |    50 | DRUG       |
|  1 | magnesium hydroxide 100mg/1ml PO. |      67 |    99 | DRUG       |
|  2 | insulin glargine                  |     180 |   195 | DRUG       |
|  3 | insulin lispro                    |     219 |   232 | DRUG       |
|  4 | metformin 1000 mg                 |     250 |   266 | DRUG       |


#### Patient Gender Classification

This model detects the gender of the patient in the clinical document. It can classify the documents into `Female`, `Male` and `Unknown`.

We release two models:

+ 'Classifierdl_gender_sbert' (more accurate, works with licensed `sbiobert_base_cased_mli`)

+ 'Classifierdl_gender_biobert' (works with `biobert_pubmed_base_cased`)

The models are trained on more than four thousands clinical documents (radiology reports, pathology reports, clinical visits etc.), annotated internally.

###### Metrics `(Classifierdl_gender_sbert)`

|        | precision | recall | f1-score | support |
| ------ | --------- | ------ | -------- | ------- |
| Female | 0.9224    | 0.8954 | 0.9087   | 239     |
| Male   | 0.7895    | 0.8468 | 0.8171   | 124     |

Text= ''*social history: shows that  does not smoke cigarettes or drink alcohol, lives in a nursing home.
family history: shows a family history of breast cancer.*''

```python
gender_classifier.annotate(text)['class'][0]
>> `Female`
```

See this [Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/21_Gender_Classifier.ipynb) notebook for further details.

a. **classifierdl_gender_sbert**

```python

document = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sbert_embedder = BertSentenceEmbeddings\
    .pretrained("sbiobert_base_cased_mli", 'en', 'clinical/models')\
    .setInputCols(["document"])\
    .setOutputCol("sentence_embeddings")\
    .setMaxSentenceLength(512)

gender_classifier = ClassifierDLModel\
    .pretrained('classifierdl_gender_sbert', 'en', 'clinical/models') \
    .setInputCols(["document", "sentence_embeddings"]) \
    .setOutputCol("class")

gender_pred_pipeline = Pipeline(
    stages = [
       document,
       sbert_embedder,
       gender_classifier
            ])
```
b. **classifierdl_gender_biobert**

```python
documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

clf_tokenizer = Tokenizer()\
    .setInputCols(["document"])\
    .setOutputCol("token")\

biobert_embeddings = BertEmbeddings().pretrained('biobert_pubmed_base_cased') \
    .setInputCols(["document",'token'])\
    .setOutputCol("bert_embeddings")

biobert_embeddings_avg = SentenceEmbeddings() \
    .setInputCols(["document", "bert_embeddings"]) \
    .setOutputCol("sentence_bert_embeddings") \
    .setPoolingStrategy("AVERAGE")

genderClassifier = ClassifierDLModel.pretrained('classifierdl_gender_biobert', 'en', 'clinical/models') \
    .setInputCols(["document", "sentence_bert_embeddings"]) \
    .setOutputCol("gender")

gender_pred_pipeline = Pipeline(
   stages = [
       documentAssembler,
       clf_tokenizer,
       biobert_embeddings,
       biobert_embeddings_avg,
       genderClassifier
   ])

   ```
#### New ICD10CM and RxCUI resolvers powered by s-Bert embeddings

The advent of s-Bert sentence embeddings changed the landscape of Clinical Entity Resolvers completely in Spark NLP. Since s-Bert is already tuned on [MedNLI](https://physionet.org/content/mednli/) (medical natural language inference) dataset, it is now capable of populating the chunk embeddings in a more precise way than before.

We now release two new resolvers:

+ `sbiobertresolve_icd10cm_augmented` (augmented with synonyms, four times richer than previous resolver accuracy:

    `73% for top-1 (exact match), 89% for top-5 (previous accuracy was 59% and 64% respectively)`

+ `sbiobertresolve_rxcui` (extract RxNorm concept unique identifiers to map with ATC or durg families)
accuracy:

    `71% for top-1 (exact match), 72% for top-5
(previous accuracy was 22% and 41% respectively)`

a. **ICD10CM augmented resolver**

Text = "*This is an 82 year old male with a history of prior tobacco use , hypertension , chronic renal insufficiency , COPD , gastritis , and TIA who initially presented to Braintree with a non-ST elevation MI and Guaiac positive stools , transferred to St . Margaret\'s Center for Women & Infants for cardiac catheterization with PTCA to mid LAD lesion complicated by hypotension and bradycardia requiring Atropine , IV fluids and transient dopamine possibly secondary to vagal reaction , subsequently transferred to CCU for close monitoring , hemodynamically stable at the time of admission to the CCU .* "

|    | chunk                       |   begin |   end | code   | term                                                     |
|---:|:----------------------------|--------:|------:|:-------|:---------------------------------------------------------|
|  0 | hypertension                |      66 |    77 | I10    | hypertension                                             |
|  1 | chronic renal insufficiency |      81 |   107 | N189   | chronic renal insufficiency                              |
|  2 | COPD                        |     111 |   114 | J449   | copd - chronic obstructive pulmonary disease             |
|  3 | gastritis                   |     118 |   126 | K2970  | gastritis                                                |
|  4 | TIA                         |     134 |   136 | S0690  | transient ischemic attack (disorder)		 |
|  5 | a non-ST elevation MI       |     180 |   200 | I219   | silent myocardial infarction (disorder)                  |
|  6 | Guaiac positive stools      |     206 |   227 | K921   | guaiac-positive stools                                   |
|  7 | mid LAD lesion              |     330 |   343 | I2102  | stemi involving left anterior descending coronary artery |
|  8 | hypotension                 |     360 |   370 | I959   | hypotension                                              |
|  9 | bradycardia                 |     376 |   386 | O9941  | bradycardia                                              |


b. **RxCUI resolver**

Text= "*He was seen by the endocrinology service and she was discharged on 50 mg of eltrombopag oral at night, 5 mg amlodipine with meals, and metformin 1000 mg two times a day .* "

|    | chunk                     |   begin |   end |   code | term                                        |
|---:|:--------------------------|--------:|------:|-------:|:--------------------------------------------|
|  0 | 50 mg of eltrombopag oral |      67 |    91 | 825427 | eltrombopag 50 MG Oral Tablet               |
|  1 | 5 mg amlodipine           |     103 |   117 | 197361 | amlodipine 5 MG Oral Tablet                 |
|  2 | metformin 1000 mg         |     135 |   151 | 861004 | metformin hydrochloride 1000 MG Oral Tablet |

Using this new resolver and some other resources like Snomed Resolver, RxTerm, MESHPA and ATC dictionary, you can link the drugs to the pharmacological actions (PA), ingredients and the disease treated with that.

###### Code sample:

(after getting the chunk from ChunkConverter)


 ```python

c2doc = Chunk2Doc().setInputCols("ner_chunk").setOutputCol("ner_chunk_doc")

sbert_embedder = BertSentenceEmbeddings\
    .pretrained("sbiobert_base_cased_mli",'en','clinical/models')\
    .setInputCols(["ner_chunk_doc"])\
    .setOutputCol("sbert_embeddings")

icd10_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_icd10cm_augmented","en", "clinical/models") \
    .setInputCols(["ner_chunk", "sbert_embeddings"]) \
    .setOutputCol("icd10cm_code")\
    .setDistanceFunction("EUCLIDEAN")
```
See the [notebook](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb#scrollTo=VtDWAlnDList) for details.


### 2.7.1

We are glad to announce that Spark NLP for Healthcare 2.7.1 has been released !

In this release, we introduce the following features:

#### 1. Sentence BioBert and Bluebert Transformers that are fine tuned on [MedNLI](https://physionet.org/content/mednli/) dataset.

Sentence Transformers offers a framework that provides an easy method to compute dense vector representations for sentences and paragraphs (also known as sentence embeddings). The models are based on BioBert and BlueBert, and are tuned specifically to meaningful sentence embeddings such that sentences with similar meanings are close in vector space. These are the first PyTorch based models we managed to port into Spark NLP.

Here is how you can load these:
```python
sbiobert_embeddins = BertSentenceEmbeddings\
     .pretrained("sbiobert_base_cased_mli",'en','clinical/models')\
     .setInputCols(["ner_chunk_doc"])\
     .setOutputCol("sbert_embeddings")
```
```python
sbluebert_embeddins = BertSentenceEmbeddings\
     .pretrained("sbluebert_base_cased_mli",'en','clinical/models')\
     .setInputCols(["ner_chunk_doc"])\
     .setOutputCol("sbert_embeddings")
```

#### 2. SentenceEntityResolvers powered by s-Bert embeddings.

The advent of s-Bert sentence embeddings changed the landscape of Clinical Entity Resolvers completely in Spark NLP. Since s-Bert is already tuned on MedNLI (medical natural language inference) dataset, it is now capable of populating the chunk embeddings in a more precise way than before.

Using sbiobert_base_cased_mli, we trained the following Clinical Entity Resolvers:

sbiobertresolve_icd10cm  
sbiobertresolve_icd10pcs  
sbiobertresolve_snomed_findings (with clinical_findings concepts from CT version)  
sbiobertresolve_snomed_findings_int  (with clinical_findings concepts from INT version)  
sbiobertresolve_snomed_auxConcepts (with Morph Abnormality, Procedure, Substance, Physical Object, Body Structure concepts from CT version)  
sbiobertresolve_snomed_auxConcepts_int  (with Morph Abnormality, Procedure, Substance, Physical Object, Body Structure concepts from INT version)  
sbiobertresolve_rxnorm  
sbiobertresolve_icdo  
sbiobertresolve_cpt  

Code sample:

(after getting the chunk from ChunkConverter)

```python
c2doc = Chunk2Doc().setInputCols("ner_chunk").setOutputCol("ner_chunk_doc")
sbert_embedder = BertSentenceEmbeddings\
     .pretrained("sbiobert_base_cased_mli",'en','clinical/models')\
     .setInputCols(["ner_chunk_doc"])\
     .setOutputCol("sbert_embeddings")

snomed_ct_resolver = SentenceEntityResolverModel
 .pretrained("sbiobertresolve_snomed_findings","en", "clinical/models") \
 .setInputCols(["ner_chunk", "sbert_embeddings"]) \
 .setOutputCol("snomed_ct_code")\
 .setDistanceFunction("EUCLIDEAN")
 ```

Output:

|    | chunks                      |   begin |   end |      code | resolutions
|  2 | COPD  				 |     113 |   116 |  13645005 | copd - chronic obstructive pulmonary disease
|  8 | PTCA                        |     324 |   327 | 373108000 | post percutaneous transluminal coronary angioplasty (finding)
| 16 | close monitoring            |     519 |   534 | 417014005 | on examination - vigilance


See the [notebook](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb#scrollTo=VtDWAlnDList) for details.

#### 3. We are releasing the following pretrained clinical NER models:

ner_drugs_large   
(trained with medications dataset, and extracts drugs with the dosage, strength, form and route at once as a single entity; entities: drug)  
ner_deid_sd_large  
(extracts PHI entities, trained with augmented dataset)  
ner_anatomy_coarse  
(trained with enriched anatomy NER dataset; entities: anatomy)  
ner_anatomy_coarse_biobert  
chunkresolve_ICD10GM_2021 (German ICD10GM resolver)


We are also releasing two new NER models:

ner_aspect_based_sentiment  
(extracts positive, negative and neutral aspects about restaurants from the written feedback given by reviewers. )  
ner_financial_contract  
(extract financial entities from contracts. See the [notebook](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/19.Financial_Contract_NER.ipynb) for details.)  


### 2.7.0

We are glad to announce that Spark NLP for Healthcare 2.7 has been released !

In this release, we introduce the following features:

#### 1. Text2SQL

Text2SQL Annotator that translates natural language text into SQL queries against a predefined database schema, which is one of the
most sought-after features of NLU. With the help of a pretrained text2SQL model, you will be able to query your database without writing a SQL query:

Example 1

Query:
What is the name of the nurse who has the most appointments?

Generated
SQL query from the model:

```sql
SELECT T1.Name  
FROM Nurse AS T1  
JOIN Appointment AS T2 ON T1.EmployeeID = T2.PrepNurse  
GROUP BY T2.prepnurse  
ORDER BY count(*) DESC  
LIMIT 1  
```

Response:  

|    | Name           |
|---:|:---------------|
|  0 | Carla Espinosa |

Example 2

Query:
How many patients do each physician take care of? List their names and number of patients they take care of.

Generated
SQL query from the model:

```sql
SELECT T1.Name,  
count(*)  
FROM Physician AS T1  
JOIN Patient AS T2 ON T1.EmployeeID = T2.PCP  
GROUP BY T1.Name  
```

Response:   

|    | Name             |   count(*) |
|---:|:-----------------|-----------:|
|  0 | Christopher Turk |          1 |
|  1 | Elliot Reid      |          2 |
|  2 | John Dorian      |          1 |


For now, it only comes with one pretrained model (trained on Spider
dataset) and new pretrained models will be released soon.

Check out the
Colab notebook to  see more examples and run on your data.


#### 2. SentenceEntityResolvers

In addition to ChunkEntityResolvers, we now release our first BioBert-based entity resolvers using the SentenceEntityResolver
annotator. It’s
fully trainable and comes with several pretrained entity resolvers for the following medical terminologies:

CPT: `biobertresolve_cpt`  
ICDO: `biobertresolve_icdo`  
ICD10CM: `biobertresolve_icd10cm`  
ICD10PCS: `biobertresolve_icd10pcs`  
LOINC: `biobertresolve_loinc`  
SNOMED_CT (findings): `biobertresolve_snomed_findings`  
SNOMED_INT (clinical_findings): `biobertresolve_snomed_findings_int`    
RXNORM (branded and clinical drugs): `biobertresolve_rxnorm_bdcd`  

Example:
```python
text = 'He has a starvation ketosis but nothing significant for dry oral mucosa'
df = get_icd10_codes (light_pipeline_icd10, 'icd10cm_code', text)
```

|    | chunks               |   begin |   end | code |
|---:|:---------------------|--------:|------:|:-------|
|  0 | a starvation ketosis |       7 |    26 | E71121 |                                                                                                                                                   
|  1 | dry oral mucosa      |      66 |    80 | K136   |


Check out the Colab notebook to  see more examples and run on your data.

You can also train your own entity resolver using any medical terminology like MedRa and UMLS. Check this notebook to
learn more about training from scratch.


#### 3. ChunkMerge Annotator

In order to use multiple NER models in the same pipeline, Spark NLP Healthcare has ChunkMerge Annotator that is used to return entities from each NER
model by overlapping. Now it has a new parameter to avoid merging overlapping entities (setMergeOverlapping)
to return all the entities regardless of char indices. It will be quite useful to analyze what every NER module returns on the same text.


#### 4. Starting SparkSession


We now support starting SparkSession with a different version of the open source jar and not only the one it was built
against by `sparknlp_jsl.start(secret, public="x.x.x")` for extreme cases.


#### 5. Biomedical NERs

We are releasing 3 new biomedical NER models trained with clinical embeddings (all one single entity models)  

`ner_bacterial_species` (comprising of Linneaus and Species800 datasets)  
`ner_chemicals` (general purpose and bio chemicals, comprising of BC4Chem and BN5CDR-Chem)  
`ner_diseases_large` (comprising of ner_disease, NCBI_Disease and BN5CDR-Disease)  

We are also releasing the biobert versions of the several clinical NER models stated below:  
`ner_clinical_biobert`  
`ner_anatomy_biobert`  
`ner_bionlp_biobert`  
`ner_cellular_biobert`  
`ner_deid_biobert`  
`ner_diseases_biobert`  
`ner_events_biobert`  
`ner_jsl_biobert`  
`ner_chemprot_biobert`  
`ner_human_phenotype_gene_biobert`  
`ner_human_phenotype_go_biobert`  
`ner_posology_biobert`  
`ner_risk_factors_biobert`  


Metrics (micro averages excluding O’s):

|    | model_name                        |   clinical_glove_micro |   biobert_micro |
|---:|:----------------------------------|-----------------------:|----------------:|
|  0 | ner_chemprot_clinical             |               0.816 |      0.803 |
|  1 | ner_bionlp                        |               0.748 |      0.808  |
|  2 | ner_deid_enriched                 |               0.934 |      0.918  |
|  3 | ner_posology                      |               0.915 |      0.911    |
|  4 | ner_events_clinical               |               0.801 |      0.809  |
|  5 | ner_clinical                      |               0.873 |      0.884  |
|  6 | ner_posology_small                |               0.941 |            |
|  7 | ner_human_phenotype_go_clinical   |               0.922 |      0.932  |
|  8 | ner_drugs                         |               0.964 |           |
|  9 | ner_human_phenotype_gene_clinical |               0.876 |      0.870  |
| 10 | ner_risk_factors                  |               0.728 |            |
| 11 | ner_cellular                      |               0.813 |      0.812  |
| 12 | ner_posology_large                |               0.921 |            |
| 13 | ner_anatomy                       |               0.851 |      0.831  |
| 14 | ner_deid_large                    |               0.942 |            |
| 15 | ner_diseases                      |               0.960 |      0.966  |


In addition to these, we release two new German NER models:

`ner_healthcare_slim` ('TIME_INFORMATION', 'MEDICAL_CONDITION',  'BODY_PART',  'TREATMENT', 'PERSON', 'BODY_PART')  
`ner_traffic` (extract entities regarding traffic accidents e.g. date, trigger, location etc.)  

#### 6. PICO Classifier

Successful evidence-based medicine (EBM) applications rely on answering clinical questions by analyzing large medical literature databases. In order to formulate
a well-defined, focused clinical question, a framework called PICO is widely used, which identifies the sentences in a given medical text that belong to the four components: Participants/Problem (P)  (e.g., diabetic patients), Intervention (I) (e.g., insulin),
Comparison (C) (e.g., placebo)  and Outcome (O) (e.g., blood glucose levels).

Spark NLP now introduces a pretrained PICO Classifier that
is trained with Biobert embeddings.

Example:

```python
text = “There appears to be no difference in smoking cessation effectiveness between 1mg and 0.5mg varenicline.”
pico_lp_pipeline.annotate(text)['class'][0]

ans: CONCLUSIONS
```

### 2.6.2

#### Overview
We are very happy to announce that version 2.6.2 of Spark NLP Enterprise is ready to be installed and used.
We are making available Named Entity Recognition, Sentence Classification and Entity Resolution models to analyze Adverse Drug Events in natural language text from clinical domains.

#### Models

##### NERs
We are pleased to announce that we have a brand new named entity recognition (NER) model for Adverse Drug Events (ADE) to extract ADE and DRUG entities from a given text.

ADE NER will have four versions in the library, trained with different size of word embeddings:

`ner_ade_bioert` (768d Bert embeddings)  
`ner_ade_clinicalbert` (768d Bert embeddings)  
`ner_ade_clinical` (200d clinical embeddings)  
`ner_ade_healthcare` (100d healthcare embeddings)  

More information and examples [here](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/16.Adverse_Drug_Event_ADE_NER_and_Classifier.ipynb)

We are also releasing our first clinical pretrained classifier for ADE classification tasks. This new ADE classifier is trained on various ADE datasets, including the mentions in tweets to represent the daily life conversations as well. So it works well on the texts coming from academic context, social media and clinical notes. It’s trained with `Clinical Biobert` embeddings, which is the most powerful contextual language model in the clinical domain out there.

##### Classifiers
ADE classifier will have two versions in the library, trained with different Bert embeddings:

`classifierdl_ade_bioert` (768d BioBert embeddings)  
`classifierdl_adee_clinicalbert` (768d ClinicalBert embeddings)  

More information and examples [here](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/16.Adverse_Drug_Event_ADE_NER_and_Classifier.ipynb)

##### Pipeline
By combining ADE NER and Classifier, we are releasing a new pretrained clinical pipeline for ADE tasks to save you from building pipelines from scratch. Pretrained pipelines are already fitted using certain annotators and transformers according to various use cases and you can use them as easy as follows:
```python
pipeline = PretrainedPipeline('explain_clinical_doc_ade', 'en', 'clinical/models')

pipeline.annotate('my string')
```
`explain_clinical_doc_ade` is bundled with `ner_ade_clinicalBert`, and `classifierdl_ade_clinicalBert`. It can extract ADE and DRUG clinical entities, and then assign ADE status to a text (`True` means ADE, `False` means not related to ADE).

More information and examples [here](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/11.Pretrained_Clinical_Pipelines.ipynb)


##### Entity Resolver
We are releasing the first Entity Resolver for `Athena` (Automated Terminology Harmonization, Extraction and Normalization for Analytics, http://athena.ohdsi.org/) to extract concept ids via standardized medical vocabularies. For now, it only supports `conditions` section and can be used to map the clinical conditions with the corresponding standard terminology and then get the concept ids to store them in various database schemas.
It is named as `chunkresolve_athena_conditions_healthcare`.

We added slim versions of several clinical NER models that are trained with 100d healthcare word embeddings, which is lighter and smaller in size.

`ner_healthcare`
`assertion_dl_healthcare`
`ner_posology_healthcare`
`ner_events_healthcare`

##### Graph Builder
Spark NLP Licensed version has several DL based annotators (modules) such as NerDL, AssertionDL, RelationExtraction and GenericClassifier, and they are all based on Tensorflow (tf) with custom graphs. In order to make the creating and customizing the tf graphs for these models easier for our licensed users, we added a graph builder to the Python side of the library. Now you can customize your graphs and use them in the respected models while training a new DL model.

```python
from sparknlp_jsl.training import tf_graph

tf_graph.build("relation_extraction",build_params={"input_dim": 6000, "output_dim": 3, 'batch_norm':1, "hidden_layers": [300, 200], "hidden_act": "relu", 'hidden_act_l2':1}, model_location=".", model_filename="re_with_BN")
```
More information and examples [here](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/17.Graph_builder_for_DL_models.ipynb)


### 2.6.0

#### Overview

We are honored to announce that Spark NLP Enterprise 2.6.0 has been released.
The first time ever, we release three pretrained clinical pipelines to save you from building pipelines from scratch. Pretrained pipelines are already fitted using certain annotators and transformers according to various use cases.
The first time ever, we are releasing 3 licensed German models for healthcare and Legal domains.

</div><div class="h3-box" markdown="1">

#### Models

##### Pretrained Pipelines:

The first time ever, we release three pretrained clinical pipelines to save you from building pipelines from scratch.
Pretrained pipelines are already fitted using certain annotators and transformers according to various use cases and you can use them as easy as follows:

```python
pipeline = PretrainedPipeline('explain_clinical_doc_carp', 'en', 'clinical/models')

pipeline.annotate('my string')
```

Pipeline descriptions:

- ```explain_clinical_doc_carp``` a pipeline with ner_clinical, assertion_dl, re_clinical and ner_posology. It will extract clinical and medication entities, assign assertion status and find relationships between clinical entities.

- ```explain_clinical_doc_era``` a pipeline with ner_clinical_events, assertion_dl and re_temporal_events_clinical. It will extract clinical entities, assign assertion status and find temporal relationships between clinical entities.

- ```recognize_entities_posology``` a pipeline with ner_posology. It will only extract medication entities.

More information and examples are available here: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/11.Pretrained_Clinical_Pipelines.ipynb.

</div><div class="h3-box" markdown="1">

#### Pretrained Named Entity Recognition and Relationship Extraction Models (English)

RE models:

```
re_temporal_events_clinical
re_temporal_events_enriched_clinical
re_human_phenotype_gene_clinical
re_drug_drug_interaction_clinical
re_chemprot_clinical
```
NER models:

```
ner_human_phenotype_gene_clinical
ner_human_phenotype_go_clinical
ner_chemprot_clinical
```
More information and examples here:
https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/10.Clinical_Relation_Extraction.ipynb

</div><div class="h3-box" markdown="1">

#### Pretrained Named Entity Recognition and Relationship Extraction Models (German)

The first time ever, we are releasing 3 licensed German models for healthcare and Legal domains.

- ```German Clinical NER``` model for 19 clinical entities

- ```German Legal NER``` model for 19 legal entities

- ```German ICD-10GM```

More information and examples here:

https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/14.German_Healthcare_Models.ipynb

https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/15.German_Legal_Model.ipynb

</div><div class="h3-box" markdown="1">

#### Other Pretrained Models

We now have Named Entity Disambiguation model out of the box.

Disambiguation models map words of interest, such as names of persons, locations and companies, from an input text document to corresponding unique entities in a target Knowledge Base (KB).

https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/12.Named_Entity_Disambiguation.ipynb

Due to ongoing requests about Clinical Entity Resolvers, we release a notebook to let you see how to train an entity resolver using an open source dataset based on Snomed.

https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/13.Snomed_Entity_Resolver_Model_Training.ipynb

</div><div class="h3-box" markdown="1">

### 2.5.5

#### Overview

We are very happy to release Spark NLP for Healthcare 2.5.5 with a new state-of-the-art RelationExtraction annotator to identify relationships between entities coming from our pretrained NER models.
This is also the first release to support Relation Extraction with the following two (2) models: `re_clinical` and `re_posology` in the `clinical/models` repository.
We also include multiple bug fixes as usual.

</div><div class="h3-box" markdown="1">

#### New Features

* RelationExtraction annotator that receives `WORD_EMBEDDINGS`, `POS`, `CHUNK`, `DEPENDENCY` and returns the CATEGORY of the relationship and a confidence score.

</div><div class="h3-box" markdown="1">

#### Enhancements

* AssertionDL Annotator now keeps logs of the metrics while training
* DeIdentification now has a default behavior of merging entities close in Levenshtein distance with `setConsistentObfuscation` and `setSameEntityThreshold` params.
* DeIdentification now has a specific parameter `setObfuscateDate` to obfuscate dates (which will be otherwise just masked). The only formats obfuscated when the param is true will be the ones present in `dateFormats` param.
* NerConverterInternal now has a `greedyMode` param that will merge all contiguous tags of the same type regardless of boundary tags like "B","E","S".
* AnnotationToolJsonReader includes `mergeOverlapping` parameter to merge (or not) overlapping entities from the Annotator jsons i.e. not included in the assertion list.

</div><div class="h3-box" markdown="1">

#### Bugfixes

* DeIdentification documentation bug fix (typo)
* DeIdentification training bug fix in obfuscation dictionary
* IOBTagger now has the correct output type `NAMED_ENTITY`

</div><div class="h3-box" markdown="1">

#### Deprecations

* EnsembleEntityResolver has been deprecated

Models

* We have 2 new `english` Relationship Extraction model for Clinical and Posology NERs:
   - `re_clinical`: with `ner_clinical` and `embeddings_clinical`
   - `re_posology`: with `ner_posology` and `embeddings_clinical`

</div><div class="h3-box" markdown="1">

### 2.5.3

#### Overview

We are pleased to announce the release of Spark NLP for Healthcare 2.5.3.
This time we include four (4) new Annotators: FeatureAssembler, GenericClassifier, Yake Keyword Extractor and NerConverterInternal.
We also include helper classes to read datasets from CodiEsp and Cantemist Spanish NER Challenges.
This is also the first release to support the following models: `ner_diag_proc` (spanish), `ner_neoplasms` (spanish), `ner_deid_enriched` (english).
We have also included Bugifxes and Enhancements for AnnotationToolJsonReader and ChunkMergeModel.

</div><div class="h3-box" markdown="1">

#### New Features

* FeatureAssembler Transformer: Receives a list of column names containing numerical arrays and concatenates them to form one single `feature_vector` annotation
* GenericClassifier Annotator: Receives a `feature_vector` annotation and outputs a `category` annotation
* Yake Keyword Extraction Annotator: Receives a `token` annotation and outputs multi-token `keyword` annotations
* NerConverterInternal Annotator: Similar to it's open source counterpart in functionality, performs smarter extraction for complex tokenizations and confidence calculation
* Readers for CodiEsp and Cantemist Challenges

</div><div class="h3-box" markdown="1">

#### Enhancements

* AnnotationToolJsonReader includes parameter for preprocessing pipeline (from Document Assembling to Tokenization)
* AnnotationToolJsonReader includes parameter to discard specific entity types

</div><div class="h3-box" markdown="1">

#### Bugfixes

* ChunkMergeModel now prioritizes highest number of different entities when coverage is the same

</div><div class="h3-box" markdown="1">

#### Models

* We have 2 new `spanish` models for Clinical Entity Recognition: `ner_diag_proc` and `ner_neoplasms`
* We have a new `english` Named Entity Recognition model for deidentification: `ner_deid_enriched`

</div><div class="h3-box" markdown="1">

### 2.5.2

#### Overview

We are really happy to bring you Spark NLP for Healthcare 2.5.2, with a couple new features and several enhancements in our existing annotators.
This release was mainly dedicated to generate adoption in our AnnotationToolJsonReader, a connector that provide out-of-the-box support for out Annotation Tool and our practices.
Also the ChunkMerge annotator has ben provided with extra functionality to remove entire entity types and to modify some chunk's entity type
We also dedicated some time in finalizing some refactorization in DeIdentification annotator, mainly improving type consistency and case insensitive entity dictionary for obfuscation.
Thanks to the community for all the feedback and suggestions, it's really comfortable to navigate together towards common functional goals that keep us agile in the SotA.

</div><div class="h3-box" markdown="1">

#### New Features

* Brand new IOBTagger Annotator
* NerDL Metrics provides an intuitive DataFrame API to calculate NER metrics at tag (token) and entity (chunk) level

</div><div class="h3-box" markdown="1">

#### Enhancements

* AnnotationToolJsonReader includes parameters for document cleanup, sentence boundaries and tokenizer split chars
* AnnotationToolJsonReader uses the task title if present and uses IOBTagger annotator
* AnnotationToolJsonReader has improved alignment in assertion train set generation by using an `alignTol` parameter as tollerance in chunk char alignment
* DeIdentification refactorization: Improved typing and replacement logic, case insensitive entities for obfuscation
* ChunkMerge Annotator now handles:
 - Drop all chunks for an entity
 - Replace entity name
 - Change entity type for a specific (chunk, entity) pair
 - Drop specific (chunk, entity) pairs
* `caseSensitive` param to EnsembleEntityResolver
* Output logs for AssertionDLApproach loss
* Disambiguator is back with improved dependency management

</div><div class="h3-box" markdown="1">

#### Bugfixes

* Bugfix in python when Annotators shared domain parts across public and internal
* Bugfix in python when ChunkMerge annotator was loaded from disk
* ChunkMerge now weights the token coverage correctly when multiple multi-token entities overlap

</div><div class="h3-box" markdown="1">

### 2.5.0

#### Overview

We are happy to bring you Spark NLP for Healthcare 2.5.0 with new Annotators, Models and Data Readers.
Model composition and iteration is now faster with readers and annotators designed for real world tasks.
We introduce ChunkMerge annotator to combine all CHUNKS extracted by different Entity Extraction Annotators.
We also introduce an Annotation Reader for JSL AI Platform's Annotation Tool.
This release is also the first one to support the models: `ner_large_clinical`, `ner_events_clinical`, `assertion_dl_large`, `chunkresolve_loinc_clinical`, `deidentify_large`
And of course we have fixed some bugs.

</div><div class="h3-box" markdown="1">

#### New Features

* AnnotationToolJsonReader is a new class that imports a JSON from AI Platform's Annotation Tool an generates NER and Assertion training datasets
* ChunkMerge Annotator is a new functionality that merges two columns of CHUNKs handling overlaps with a very straightforward logic: max coverage, max # entities
* ChunkMerge Annotator handles inputs from NerDLModel, RegexMatcher, ContextualParser, TextMatcher
* A DeIdentification pretrained model can now work in 'mask' or 'obfuscate' mode

</div><div class="h3-box" markdown="1">

#### Enhancements

* DeIdentification Annotator has a more consistent API:
    * `mode` param with values ('mask'l'obfuscate') to drive its behavior
    * `dateFormats` param a list of string values to to select which `dateFormats` to obfuscate (and which to just mask)
* DeIdentification Annotator no longer automatically obfuscates dates. Obfuscation is now driven by `mode` and `dateFormats` params
* A DeIdentification pretrained model can now work in 'mask' or 'obfuscate' mode

</div><div class="h3-box" markdown="1">

#### Bugfixes

* DeIdentification Annotator now correctly deduplicates protected entities coming from NER / Regex
* DeIdentification Annotator now indexes chunks correctly after merging them
* AssertionDLApproach Annotator can now be trained with the graph in any folder specified by setting `graphFolder` param
* AssertionDLApproach now has the `setClasses` param setter in Python wrapper
* JVM Memory and Kryo Max Buffer size increased to 32G and 2000M respectively in `sparknlp_jsl.start(secret)` function

</div><div class="h3-box" markdown="1">

### 2.4.6

#### Overview

We release Spark NLP for Healthcare 2.4.6 to fix some minor bugs.

</div><div class="h3-box" markdown="1">

#### Bugfixes

* Updated IDF value calculation to be probabilistic based log[(N - df_t) / df_t + 1] as opposed to log[N / df_t]
* TFIDF cosine distance was being calculated with the rooted norms rather than with the original squared norms
* Validation of label cols is now performed at the beginning of EnsembleEntityResolver
* Environment Variable for License value named jsl.settings.license
* Now DocumentLogRegClassifier can be serialized from Python (bug introduced with the implementation of RecursivePipelines, LazyAnnotator attribute)

</div><div class="h3-box" markdown="1">

### 2.4.5

#### Overview

We are glad to announce Spark NLP for Healthcare 2.4.5. As a new feature we are happy to introduce our new EnsembleEntityResolver which allows our Entity Resolution architecture to scale up in multiple orders of magnitude and handle datasets of millions of records on a sub-log computation increase
We also enhanced our ChunkEntityResolverModel with 5 new distance calculations with weighting-array and aggregation-strategy params that results in more levers to finetune its performance against a given dataset.

</div><div class="h3-box" markdown="1">

#### New Features

* EnsembleEntityResolver consisting of an integrated TFIDF-Logreg classifier in the first layer + Multiple ChunkEntityResolvers in the second layer (one per each class)
* Five (5) new distances calculations for ChunkEntityResolver, namely:
    - Token Based: TFIDF-Cosine, Jaccard, SorensenDice
    - Character Based: JaroWinkler and Levenshtein
* Weight parameter that works as a multiplier for each distance result to be considered during their aggregation
* Three (3) aggregation strategies for the enabled distance in a particular instance, namely: AVERAGE, MAX and MIN

</div><div class="h3-box" markdown="1">

#### Enhancements

* ChunkEntityResolver can now compute distances over all the `neighbours` found and return the metadata just for the best `alternatives` that meet the `threshold`;
before it would calculate them over the neighbours and return them all in the metadata
* ChunkEntityResolver now has an `extramassPenalty` parameter to accoun for penalization of token-length difference in compared strings
* Metadata for the ChunkEntityResolver has been updated accordingly to reflect all new features
* StringDistances class has been included in utils to aid in the calculation and organization of different types of distances for Strings
* HasFeaturesJsl trait has been included to support the serialization of Features including [T] <: AnnotatorModel[T] types

</div><div class="h3-box" markdown="1">

#### Bugfixes

* Frequency calculation for WMD in ChunkEntityResolver has been adjusted to account for real word count representation
* AnnotatorType for DocumentLogRegClassifier has been changed to CATEGORY to align with classifiers in Open Source library

</div><div class="h3-box" markdown="1">

#### Deprecations

* Legacy EntityResolver{Approach, Model} classes have been deprecated in favor of ChunkEntityResolver classes
* ChunkEntityResolverSelector classes has been deprecated in favor of EnsembleEntityResolver

</div><div class="h3-box" markdown="1">

### 2.4.2

#### Overview

We are glad to announce Spark NLP for Healthcare 2.4.2. As a new feature we are happy to introduce our new Disambiguation Annotator,
which will let the users resolve different kind of entities based on Knowledge bases provided in the form of Records in a RocksDB database.
We also enhanced / fixed DocumentLogRegClassifier, ChunkEntityResolverModel and ChunkEntityResolverSelector Annotators.

</div><div class="h3-box" markdown="1">

#### New Features

* Disambiguation Annotator (NerDisambiguator and NerDisambiguatorModel) which accepts annotator types CHUNK and SENTENCE_EMBEDDINGS and
returns DISAMBIGUATION annotator type. This output annotation type includes all the matches in the result and their similarity scores in the metadata.

</div><div class="h3-box" markdown="1">

#### Enhancements

* ChunkEntityResolver Annotator now supports both EUCLIDEAN and COSINE distance for the KNN search and WMD calculation.

</div><div class="h3-box" markdown="1">

#### Bugfixes

* Fixed a bug in DocumentLogRegClassifier Annotator to support its serialization to disk.
* Fixed a bug in ChunkEntityResolverSelector Annotator to group by both SENTENCE and CHUNK at the time of forwarding tokens and embeddings to the lazy annotators.
* Fixed a bug in ChunkEntityResolverModel in which the same exact embeddings was not included in the neighbours.

</div><div class="h3-box" markdown="1">

### 2.4.1

#### Overview

Introducing Spark NLP for Healthcare 2.4.1 after all the feedback we received in the form of issues and suggestions on our different communication channels.
Even though 2.4.0 was very stable, version 2.4.1 is here to address minor bug fixes that we summarize in the following lines.

</div><div class="h3-box" markdown="1">

#### Bugfixes

* Changing the license Spark property key to be "jsl" instead of "sparkjsl" as the latter generates inconsistencies
* Fix the alignment logic for tokens and chunks in the ChunkEntityResolverSelector because when tokens and chunks did not have the same begin-end indexes the resolution was not executed

</div><div class="h3-box" markdown="1">

### 2.4.0

#### Overview

We are glad to announce Spark NLP for Healthcare 2.4.0. This is an important release because of several refactorizations achieved in the core library, plus the introduction of several state of the art algorithms, new features and enhancements.
We have included several architecture and performance improvements, that aim towards making the library more robust in terms of storage handling for Big Data.
In the NLP aspect, we have introduced a ContextualParser, DocumentLogRegClassifier and a ChunkEntityResolverSelector.
These last two Annotators also target performance time and memory consumption by lowering the order of computation and data loaded to memory in each step when designed following a hierarchical pattern.
We have put a big effort on this one, so please enjoy and share your comments. Your words are always welcome through all our different channels.
Thank you very much for your important doubts, bug reports and feedback; they are always welcome and much appreciated.

</div><div class="h3-box" markdown="1">

#### New Features

* BigChunkEntityResolver Annotator: New experimental approach to reduce memory consumption at expense of disk IO.
* ContextualParser Annotator: New entity parser that works based on context parameters defined in a JSON file.
* ChunkEntityResolverSelector Annotator: New AnnotatorModel that takes advantage of the RecursivePipelineModel + LazyAnnotator pattern to annotate with different LazyAnnotators at runtime.
* DocumentLogregClassifier Annotator: New Annotator that provides a wrapped TFIDF Vectorizer + LogReg Classifier for TOKEN AnnotatorTypes (either at Document level or Chunk level)

</div><div class="h3-box" markdown="1">

#### Enhancements

* `normalizedColumn` Param is no longer required in ChunkEntityResolver Annotator (defaults to the `labelCol` Param value).
* ChunkEntityResolverMetadata now has more data to infer whether the match is meaningful or not.

</div><div class="h3-box" markdown="1">

#### Bugfixes

* Fixed a bug on ContextSpellChecker Annotator where unrecognized tokens would cause an exception if not in vocabulary.
* Fixed a bug on ChunkEntityResolver Annotator where undetermined results were coming out of negligible confidence scores for matches.
* Fixed a bug on ChunkEntityResolver Annotator where search would fail if the `neighbours` Param was grater than the number of nodes in the tree. Now it returns up to the number of nodes in the tree.

</div><div class="h3-box" markdown="1">

#### Deprecations

* OCR Moves to its own JSL Spark OCR project.

</div>

#### Infrastructure

* Spark NLP License is now required to utilize the library. Please follow the instructions on the shared email.
