---
layout: docs
header: true
seotitle: Spark NLP for Healthcare | John Snow Labs
title: Spark NLP for Healthcare Release Notes 2.7.2
permalink: /docs/en/spark_nlp_healthcare_versions/release_notes_2_7_2
key: docs-licensed-release-notes
modify_date: 2021-07-14
show_nav: true
sidebar:
    nav: sparknlp-healthcare
---

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


<div class="prev_ver h3-box" markdown="1">

## Versions

</div>

<ul class="pagination">
    <li>
        <a href="release_notes_2_7_1">Version 2.7.1</a>
    </li>
    <li>
        <strong>Version 2.7.2</strong>
    </li>
    <li>
        <a href="release_notes_2_7_3">Version 2.7.3</a>
    </li>
</ul>

<ul class="pagination owl-carousel pagination_big">
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
    <li class="active"><a href="release_notes_2_7_2">2.7.2</a></li>
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