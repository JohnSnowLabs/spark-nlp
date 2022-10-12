---
layout: docs
header: true
seotitle: Spark NLP for Healthcare | John Snow Labs
title: Spark NLP for Healthcare Release Notes 4.0.2
permalink: /docs/en/spark_nlp_healthcare_versions/release_notes_4_0_2
key: docs-licensed-release-notes
modify_date: 2021-07-14
show_nav: true
sidebar:
    nav: sparknlp-healthcare
---

## 4.0.2

#### Highlights

+ 16 new text classification models for English and Spanish social media text related to public health topics (stress, domestic violence, vaccine status, drug reviews etc.)
+ Pretrained medication NER pipeline to augment posology NER models with Drugbank dataset
+ Pretrained medication resolver pipeline to extract RxNorm, UMLS, NDC, SNOMED CT codes and action/treatments.
+ New disease NER model for Spanish language
+ 5 new chunk mapper models to convert clinical entities to relevant medical terminology (UMLS)
+ 5 new pretrained resolver pipelines to convert clinical entities to relevant medical terminology (UMLS)
+ New Relation Extraction model to detect Drug and ADE relations
+ New module for converting Annotation Lab (ALAB) exports into formats suitable for training new models
+ Updated De-identification pretrained pipelines
+ New `setBlackList()` parameter in `ChunkFilterer()` annotator
+ New `Doc2ChunkInternal()` annotator
+ Listing clinical pretrained models and pipelines with one-liner
+ Bug fixes
+ New and updated notebooks
+ List of recently updated or added models **(40+ new models and pipelines)**


#### 16 New Classification Models for English and Spanish Social Media Texts Related to Public Health Topics (Stress, Domestic Violence, Vaccine Status, Drug Reviews etc.)

+ We are releasing 11 new `MedicalBertForSequenceClassification` models to classify text from social media data for English and Spanish languages.

| model name                                                                                                                                                                    | description                                                                             | predicted entities                                         |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|------------------------------------------------------------|
|[bert_sequence_classifier_ade_augmented](https://nlp.johnsnowlabs.com/2022/07/27/bert_sequence_classifier_ade_augmented_en_3_0.html)                                           | this model classify tweets reporting ADEs (Adverse Drug Events).                        | `ADE` `noADE`                                              |
|[bert_sequence_classifier_health_mandates_stance_tweet](https://nlp.johnsnowlabs.com/2022/07/28/bert_sequence_classifier_health_mandates_stance_tweet_en_3_0.html)             | this model classifies stance in tweets about health mandates.                             | `FAVOR` `AGAINST` `NONE`                                   |
|[bert_sequence_classifier_health_mandates_premise_tweet](https://nlp.johnsnowlabs.com/2022/07/29/bert_sequence_classifier_health_mandates_premise_tweet_en_3_0.html)           | this model classifies premise in tweets about health mandates.                            | `has_premse` `has_no_premse`                               |
|[bert_sequence_classifier_treatement_changes_sentiment_tweet](https://nlp.johnsnowlabs.com/2022/07/28/bert_sequence_classifier_treatement_changes_sentiment_tweet_en_3_0.html) | this model classifies treatment changes reviews in tweets as `negative` and `positive`.  | `positive` `negative`                                      |
|[bert_sequence_classifier_drug_reviews_webmd](https://nlp.johnsnowlabs.com/2022/07/28/bert_sequence_classifier_drug_reviews_webmd_en_3_0.html)                                 | this model classifies drug reviews from WebMD as `negative` and `positive`.               | `positive` `negative`                                      |
|[bert_sequence_classifier_self_reported_age_tweet](https://nlp.johnsnowlabs.com/2022/07/26/bert_sequence_classifier_self_reported_age_tweet_en_3_0.html)                       | this model classifies if there is a self-reported age in social media data.                   | `self_report_age` `no_report`                              |
|[bert_sequence_classifier_self_reported_symptoms_tweet](https://nlp.johnsnowlabs.com/2022/07/28/bert_sequence_classifier_self_reported_symptoms_tweet_es_3_0.html)             | this model classifies self-reported COVID-19 symptoms in Spanish language tweets.         | `Lit-News_mentions` `Self_reports non_personal_reports`    |
|[bert_sequence_classifier_self_reported_vaccine_status_tweet](https://nlp.johnsnowlabs.com/2022/07/29/bert_sequence_classifier_self_reported_vaccine_status_tweet_en_3_0.html) | this model classifies self-reported COVID-19 vaccination status in tweets.                | `Vaccine_chatter` `Self_reports`                           |
|[bert_sequence_classifier_self_reported_partner_violence_tweet](https://nlp.johnsnowlabs.com/2022/07/28/bert_sequence_classifier_self_reported_partner_violence_tweet_en_3_0.html)| this model classifies self-reported Intimate partner violence (IPV) in tweets.          | `intimate_partner_violence` `non_intimate_partner_violence`|
|[bert_sequence_classifier_exact_age_reddit](https://nlp.johnsnowlabs.com/2022/07/26/bert_sequence_classifier_exact_age_reddit_en_3_0.html )                                    | this model classifies if there is a self-reported age in social media forum posts (Reddit).   | `self_report_age` `no_report`                   |
|[bert_sequence_classifier_self_reported_stress_tweet](https://nlp.johnsnowlabs.com/2022/07/29/bert_sequence_classifier_self_reported_stress_tweet_en_3_0.html )                | this model classifies stress in social media (Twitter) posts in the self-disclosure category.| `stressed` `not-stressed`                   |


*Example* :

```python
...
sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_exact_age_reddit", "en", "clinical/models")\
    .setInputCols(["document", "token"])\
    .setOutputCol("class")\
...
sample_text = ["Is it bad for a 19 year old it's been getting worser.",
               "I was about 10. So not quite as young as you but young."]
```

*Results* :

```bash
+-------------------------------------------------------+-----------------+
|text                                                   |class            |
+-------------------------------------------------------+-----------------+
|Is it bad for a 19 year old its been getting worser.   |[self_report_age]|
|I was about 10. So not quite as young as you but young.|[no_report]      |
+-------------------------------------------------------+-----------------+
```

+ We are releasing 5 new public health classification models.

| model name                                                                                                                                | description                                                                             | predicted entities             |
|-------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|--------------------------------|
|[bert_sequence_classifier_health_mentions](https://nlp.johnsnowlabs.com/2022/07/25/bert_sequence_classifier_health_mentions_en_3_0.html)    | This model can classify public health mentions in social media text | `figurative_mention` `other_mention` `health_mention`|
|[classifierdl_health_mentions](https://nlp.johnsnowlabs.com/2022/07/25/classifierdl_health_mentions_en_3_0.html)                           | This model can classify public health mentions in social media text | `figurative_mention` `other_mention` `health_mention`|
|[bert_sequence_classifier_vaccine_sentiment](https://nlp.johnsnowlabs.com/2022/07/28/bert_sequence_classifier_vaccine_sentiment_en_3_0.html)| This model can extract information from COVID-19 Vaccine-related tweets | `neutral` `positive` `negative` |
|[classifierdl_vaccine_sentiment](https://nlp.johnsnowlabs.com/2022/07/28/classifierdl_vaccine_sentiment_en_3_0.html)                       |  This model can extract information from COVID-19 Vaccine-related tweets | `neutral` `positive` `negative` |
|[bert_sequence_classifier_stressor](https://nlp.johnsnowlabs.com/2022/07/27/bert_sequence_classifier_stressor_en_3_0.html)                 | This model can classify source of emotional stress in text. | `Family_Issues` `Financial_Problem` `Health_Fatigue_or_Physical Pain` `Other` `School` `Work` `Social_Relationships`|


*Example* :

```python
...
sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_health_mentions", "en", "clinical/models")\
     .setInputCols(["document","token"])\
     .setOutputCol("class")
...

sample_text =["Another uncle of mine had a heart attack and passed away. Will be cremated Saturday I think I ve gone numb again RIP Uncle Mike",
              "I don't wanna fall in love. If I ever did that, I think I'd have a heart attack",
              "Aluminum is a light metal that causes dementia and Alzheimer's disease. You should never put aluminum into your body (including deodorants)."]
```

*Results* :

```bash
+--------------------------------------------------------------------------------------------------------------------------------------------+--------------------+
|text                                                                                                                                        |result              |
+--------------------------------------------------------------------------------------------------------------------------------------------+--------------------+
|Another uncle of mine had a heart attack and passed away. Will be cremated Saturday I think I ve gone numb again RIP Uncle Mike             |[health_mention]    |
|I don't wanna fall in love. If I ever did that, I think I'd have a heart attack                                                             |[figurative_mention]|
|Aluminum is a light metal that causes dementia and Alzheimer's disease. You should never put aluminum into your body (including deodorants).|[other_mention]     |
+--------------------------------------------------------------------------------------------------------------------------------------------+--------------------+
```


#### Pretrained Medication NER Pipeline to Augmented Posology NER Models with Drugbank Dataset

We are releasing a medication NER pretrained pipeline to extract medications in clinical text. It's an augmented version of posology NER model with Drugbank datasets and can retun all the medications with a single line of code without building a pipeline with models.

+ `ner_medication_pipeline`: This pretrained pipeline can detect medication entities and label them as `DRUG` in clinical text.

See [Models Hub Page](https://nlp.johnsnowlabs.com/2022/07/26/ner_medication_pipeline_en_3_0.html) for more details.

*Example* :

```python
from sparknlp.pretrained import PretrainedPipeline

medication_pipeline = PretrainedPipeline("ner_medication_pipeline", "en", "clinical/models")

text = """The patient was prescribed metformin 1000 MG, and glipizide 2.5 MG. The other patient was given Fragmin 5000 units, Xenaderm to wounds topically b.i.d. and OxyContin 30 mg."""
```

*Results* :

```bash
|--------------------|-----------|
| chunk              | ner_label |
|--------------------|-----------|
| metformin 1000 MG  | DRUG      |
| glipizide 2.5 MG   | DRUG      |
| Fragmin 5000 units | DRUG      |
| Xenaderm           | DRUG      |
| OxyContin 30 mg    | DRUG      |
|--------------------|-----------|
```

#### Pretrained Medication Resolver Pipeline to Extract RxNorm, UMLS, NDC , SNOMED CT Codes and Action/Treatments

We are releasing a medication resolver pipeline to extract medications and and resolve RxNorm, UMLS, NDC, SNOMED CT codes and action/treatments in clinical text. You can get those codes if available with a single line of code without building a pipeline with models.

+ `medication_resolver_pipeline`: This pretrained pipeline can detect medication entities and resolve codes if available.


*Example* :

```python
from sparknlp.pretrained import PretrainedPipeline

medication_pipeline = PretrainedPipeline("medication_resolver_pipeline", "en", "clinical/models")

text = """The patient was prescribed Mycobutn 150 MG, Salagen 5 MG oral tablet,
The other patient is given Lescol 40 MG and Lidoderm 0.05 MG/MG, triazolam 0.125 MG Oral Tablet, metformin hydrochloride 1000 MG Oral Tablet"""
```

*Results* :

```bash
|---------------------------------------------|----------------|---------------------|--------------------------------------------|----------|-------------|---------------|---------------|----------|
| ner_chunk                                   |   RxNorm_Chunk | Action              | Treatment                                  | UMLS     | SNOMED_CT   | NDC_Product   | NDC_Package   | entity   |
|---------------------------------------------|----------------|---------------------|--------------------------------------------|----------|-------------|---------------|---------------|----------|
| Mycobutn 150 MG                             |         103899 | Antimiycobacterials | Infection                                  | C0353536 | -           | 00013-5301    | 00013-5301-17 | DRUG     |
| Salagen 5 MG oral tablet                    |        1000915 | Antiglaucomatous    | Cancer                                     | C0361693 | -           | 59212-0705    | 59212-0705-10 | DRUG     |
| Lescol 40 MG                                |         103919 | Hypocholesterolemic | Heterozygous Familial Hypercholesterolemia | C0353573 | -           | 00078-0234    | 00078-0234-05 | DRUG     |
| Lidoderm 0.05 MG/MG                         |        1011705 | Anesthetic          | Pain                                       | C0875706 | -           | 00247-2129    | 00247-2129-30 | DRUG     |
| triazolam 0.125 MG Oral Tablet              |         198317 | -                   | -                                          | C0690642 | 373981005   | 00054-4858    | 00054-4858-25 | DRUG     |
| metformin hydrochloride 1000 MG Oral Tablet |         861004 | -                   | -                                          | C0978482 | 376701008   | 00093-7214    | 00185-0221-01 | DRUG     |
|---------------------------------------------|----------------|---------------------|--------------------------------------------|----------|-------------|---------------|---------------|----------|
```


#### New Disease NER Model for Spanish Language

We are releasing a new `MedicalBertForTokenClassifier` model to extract disease entities from social media text in Spanish.

+ `bert_token_classifier_disease_mentions_tweet`: This model can extract disease entities in Spanish tweets and label them as `ENFERMEDAD` (disease).

See [Models Hub Page](https://nlp.johnsnowlabs.com/2022/07/28/bert_token_classifier_disease_mentions_tweet_es_3_0.html) for more details.

*Example* :

```python
...
tokenClassifier = MedicalBertForTokenClassifier.pretrained("bert_token_classifier_disease_mentions_tweet", "es", "clinical/models")\
  .setInputCols("token", "sentence")\
  .setOutputCol("label")\
  .setCaseSensitive(True)
...
example_text = """El diagnóstico fueron varios. Principal: Neumonía en el pulmón derecho. Sinusitis de caballo, Faringitis aguda e infección de orina, también elevada. Gripe No. Estuvo hablando conmigo, sin exagerar, mas de media hora, dándome ánimo y fuerza y que sabe, porque ha visto"""
```

*Results* :

```bash
+---------------------+----------+
|chunk                |ner_label |
+---------------------+----------+
|Neumonía en el pulmón|ENFERMEDAD|
|Sinusitis            |ENFERMEDAD|
|Faringitis aguda     |ENFERMEDAD|
|infección de orina   |ENFERMEDAD|
|Gripe                |ENFERMEDAD|
+---------------------+----------+
```


#### 5 new Chunk Mapper Models to Convert Clinical Entities to Relevant Medical Terminology (UMLS)

We are releasing 5 new `ChunkMapperModel` models to map clinical entities with their corresponding UMLS CUI codes.

| Mapper Name                                                                                                      | Source                  | Target   |
|--------------------------------------------------------------------------------------------------------------------|-------------------------|----------|
| [umls_clinical_drugs_mapper](https://nlp.johnsnowlabs.com/2022/07/06/umls_clinical_drugs_mapper_en_3_0.html)       | Drugs                   | UMLS CUI |
| [umls_clinical_findings_mapper](https://nlp.johnsnowlabs.com/2022/07/08/umls_clinical_findings_mapper_en_3_0.html) | Clinical Findings       | UMLS CUI |
| [umls_disease_syndrome_mapper](https://nlp.johnsnowlabs.com/2022/07/11/umls_disease_syndrome_mapper_en_3_0.html)   | Disease and Syndromes   | UMLS CUI |
| [umls_major_concepts_mapper](https://nlp.johnsnowlabs.com/2022/07/11/umls_major_concepts_mapper_en_3_0.html)       | Clinical Major Concepts | UMLS CUI |
| [umls_drug_substance_mapper](https://nlp.johnsnowlabs.com/2022/07/11/umls_drug_substance_mapper_en_3_0.html)       | Drug Substances         | UMLS CUI |

*Example* :

```python
...
ner_model = MedicalNerModel.pretrained("ner_posology_greedy", "en", "clinical/models")\
     .setInputCols(["sentence", "token", "embeddings"])\
     .setOutputCol("clinical_ner")

ner_model_converter = NerConverterInternal()\
     .setInputCols("sentence", "token", "clinical_ner")\
     .setOutputCol("ner_chunk")

chunkerMapper = ChunkMapperModel.pretrained("umls_drug_substance_mapper", "en", "clinical/models")\
       .setInputCols(["ner_chunk"])\
       .setOutputCol("mappings")\
       .setRels(["umls_code"])\
       .setLowerCase(True)
...

example_text = """The patient was given  metformin, lenvatinib and lavender 700 ml/ml"""
```

*Results* :

```bash
+------------------+---------+---------+
|         ner_chunk|ner_label|umls_code|
+------------------+---------+---------+
|         metformin|     DRUG| C0025598|
|        lenvatinib|     DRUG| C2986924|
|lavender 700 ml/ml|     DRUG| C0772360|
+------------------+---------+---------+
```

#### 5 new Pretrained Resolver Pipelines to Convert Clinical Entities to Relevant Medical Terminology (UMLS)

We now have 5 new resolver `PretrainedPipeline` to convert clinical entities to their UMLS CUI codes. You just need to feed your text and it will return the corresponding UMLS codes.

| Pipeline Name                                                                                                                            | Entity                  | Target   |
|------------------------------------------------------------------------------------------------------------------------------------------|-------------------------|----------|
| [umls_drug_resolver_pipeline](https://nlp.johnsnowlabs.com/2022/07/26/umls_drug_resolver_pipeline_en_3_0.html)                           | Drugs                   | UMLS CUI |
| [umls_clinical_findings_resolver_pipeline](https://nlp.johnsnowlabs.com/2022/07/26/umls_clinical_findings_resolver_pipeline_en_3_0.html) | Clinical Findings       | UMLS CUI |
| [umls_disease_syndrome_resolver_pipeline](https://nlp.johnsnowlabs.com/2022/07/26/umls_disease_syndrome_resolver_pipeline_en_3_0.html)   | Disease and Syndromes   | UMLS CUI |
| [umls_major_concepts_resolver_pipeline](https://nlp.johnsnowlabs.com/2022/07/25/umls_major_concepts_resolver_pipeline_en_3_0.html)       | Clinical Major Concepts | UMLS CUI |
| [umls_drug_substance_resolver_pipeline](https://nlp.johnsnowlabs.com/2022/07/25/umls_drug_substance_resolver_pipeline_en_3_0.html)       | Drug Substances         | UMLS CUI |

*Example* :

```python
from sparknlp.pretrained import PretrainedPipeline

pipeline= PretrainedPipeline("umls_clinical_findings_resolver_pipeline", "en", "clinical/models")

sample_text = "HTG-induced pancreatitis associated with an acute hepatitis, and obesity"
```

*Results* :

```bash
+-------------------------+---------+---------+
|chunk                    |ner_label|umls_code|
+-------------------------+---------+---------+
|HTG-induced pancreatitis |PROBLEM  |C1963198 |
|an acute hepatitis       |PROBLEM  |C4750596 |
|obesity                  |PROBLEM  |C1963185 |
+-------------------------+---------+---------+
```


#### New Relation Extraction Model to Detect Drug and ADE relations

We are releasing new `re_ade_conversational` model that can extract relations between `DRUG` and `ADE` entities from conversational texts and tag the relations as `is_related` and `not_related`.

See [Models Hub Page](https://nlp.johnsnowlabs.com/2022/07/27/re_ade_conversational_en_3_0.html) for more details.

*Example* :

```python
...
re_model = RelationExtractionModel().pretrained("re_ade_conversational", "en", 'clinical/models')\
    .setInputCols(["embeddings", "pos_tags", "ner_chunks", "dependencies"])\
    .setOutputCol("relations")\
    .setRelationPairs(["ade-drug", "drug-ade"])
...

sample_text = "E19.32 day 20 rivaroxaban diary. still residual aches and pains; only had 4 paracetamol today."
```

*Results* :

```bash
|--------------------------|----------|-------------|---------|-------------|
| chunk1                   | entitiy1 | chunk2      | entity2 | relation    |
|--------------------------|----------|-------------|---------|-------------|
| residual aches and pains | ADE      | rivaroxaban | DRUG    | is_related  |
| residual aches and pains | ADE      | paracetamol | DRUG    | not_related |
|--------------------------|----------|-------------|---------|-------------|
```

#### New Module for Converting Annotation Lab (ALAB) Exports Into Suitable Formats for Training New Models

 We have a new `sparknlp_jsl.alab` module with functions for converting ALAB JSON exports into suitable formats for training NER, Assertion and Relation Extraction models.

 *Example* :

```python
from sparknlp_jsl.alab import get_conll_data, get_assertion_data, get_relation_extraction_data

get_conll_data(spark=spark, input_json_path="alab_demo.json", output_name="conll_demo")

assertion_df = get_assertion_data(spark=spark, input_json_path = 'alab_demo.json', assertion_labels = ['ABSENT'], relevant_ner_labels = ['PROBLEM', 'TREATMENT'])

relation_df = get_relation_extraction_data(spark=spark, input_json_path='alab_demo.json')
```

These functions contain over 10 arguments each which give you all the flexibility you need to convert your annotations to trainable formats. These include parameters controlling tokenization, ground truth selections, negative annotations, negative annotation weights, task exclusions, and many more. To find out how to make best use of these functions, head over to [this repository](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Annotation_Lab).

#### Updated De-identification Pretrained Pipelines

 We have updated de-identification pretrained pipelines to provide better performance than ever before. This includes an update to the `clinical_deidentification` pretrained pipeline and a new light-weight version `clinical_deidentification_slim`.

*Example* :

```python
from sparknlp.pretrained import PretrainedPipeline

deid_pipeline = PretrainedPipeline("clinical_deidentification", "en", "clinical/models")
slim_deid_pipeline = PretrainedPipeline("clinical_deidentification_slim", "en", "clinical/models")

sample_text = "Name : Hendrickson, Ora, Record date: 2093-01-13, # 719435"
```

*Results* :

```bash
Name : <PATIENT>, Record date: <DATE>, <MEDICALRECORD>
Name : [**************], Record date: [********], [****]
Name : ****, Record date: ****, ****
Name : Alexia Mcgill, Record date: 2093-02-19, Y138038
```

#### New `setBlackList()` Parameter in `ChunkFilterer()` Annotator

We are releasing a new `setBlackList()` parameter in the `ChunkFilterer()` annotator. `ChunkFilterer()` lets through every chunk except those that match the list of phrases or regex rules in the `setBlackList()` parameter.

*Example* :

```python
...
chunk_filterer = ChunkFilterer()\
    .setInputCols("sentence","ner_chunk")\
    .setOutputCol("chunk_filtered")\
    .setCriteria("isin")\
    .setBlackList(['severe fever', 'severe cough'])
...

example_text= """Patient with severe fever, severe cough, sore throat, stomach pain, and a headache."""
```

*Results* :

```bash
+-------------------------------------------------------------------+---------------------------------------+
|ner_chunk                                                          |chunk_filtered                         |
+-------------------------------------------------------------------+---------------------------------------+
|[severe fever, severe cough, sore throat, stomach pain, a headache]|[sore throat, stomach pain, a headache]|
+-------------------------------------------------------------------+---------------------------------------+
```

#### New `Doc2ChunkInternal()` Annotator

We are releasing a `Doc2ChunkInternal()` annotator. This is a licensed version of the open source `Doc2Chunk()` annotator. You can now customize the tokenization step within `Doc2Chunk()`. This will be quite handy when it comes to training custom assertion models.

*Example* :

```python
...
doc2ChunkInternal = Doc2ChunkInternal()\
.setInputCols("document","token")\
.setStartCol("start")\
.setChunkCol("target")\
.setOutputCol("doc2chunkInternal")

...

df= spark.createDataFrame([
    ["The mass measures 4 x 3.5cm in size more.",8,"size"],
    ["The mass measures 4 x 3.5cm in size more.",9,"size"]]).toDF("sentence","start", "target")

```

*Results* :

```bash
+-----------------------------------------+-----+------+--------------------------------------------------------+-----------------------------------------------------------+
|                                 sentence|start|target|                                       doc2chunkInternal|                                                  doc2chunk|
+-----------------------------------------+-----+------+--------------------------------------------------------+-----------------------------------------------------------+
|The mass measures 4 x 3.5cm in size more.|    8|  size|[{chunk, 31, 34, size, {sentence -> 0, chunk -> 0}, []}]|[{chunk, 31, 34, size, {sentence -> 0, chunk -> 0}, []}]   |
|The mass measures 4 x 3.5cm in size more.|    9|  size|                                                      []|[{chunk, 31, 34, size, {sentence -> 0, chunk -> 0}, []}]   |
+-----------------------------------------+-----+------+--------------------------------------------------------+-----------------------------------------------------------+
```


#### Listing Pretrained Clinical Models and Pipelines with One-Liner

We have new `returnPrivatePipelines()` and `returnPrivateModels()` features under `InternalResourceDownloader` package to return licensed models and pretrained pipelines as a list.

*Example* :

```python
from sparknlp_jsl.pretrained import InternalResourceDownloader

# pipelines = InternalResourceDownloader.returnPrivatePipelines()
assertion_models = InternalResourceDownloader.returnPrivateModels("AssertionDLModel")
```

*Results* :

```bash
[['assertion_ml', 'en', '2.0.2'],
 ['assertion_dl', 'en', '2.0.2'],
 ['assertion_dl_healthcare', 'en', '2.7.2'],
 ['assertion_dl_biobert', 'en', '2.7.2'],
 ['assertion_dl', 'en', '2.7.2'],
 ['assertion_dl_radiology', 'en', '2.7.4'],
 ['assertion_jsl_large', 'en', '3.1.2'],
 ['assertion_jsl', 'en', '3.1.2'],
 ['assertion_dl_scope_L10R10', 'en', '3.4.2'],
 ['assertion_dl_biobert_scope_L10R10', 'en', '3.4.2'],
 ['assertion_oncology_treatment_binary_wip', 'en', '3.5.0']]
```

#### Bug Fixes
+ `ZeroShotRelationExtractionModel`: Fixed the issue that blocks the use of this annotator.
+ `AnnotationToolJsonReader`: Fixed the issue with custom pipeline usage in this annotator.
+ `RelationExtractionApproach`: Fixed issues related to training logs and inference.

#### New and Updated Notebooks
+ [Clinical Named Entity Recognition Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb): Added new `getPrivateModel()` feature
+ [Clinical Entity Resolvers Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb): Added an example of **reseolver pretrained pipelines**
+ [Pretrained Clinical Pipelines Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/11.Pretrained_Clinical_Pipelines.ipynb): Pipeline list updated and examples of **resolver pretrained pipelines** were added
+ [Chunk Mapping Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/26.Chunk_Mapping.ipynb): New mapper models added into model list
+ All [certification notebooks](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings/Healthcare) updated with v4.0.0.

#### List of Recently Updated and Added Models and Pretrained Pipelines
- `bert_token_classifier_ner_anatem`
- `bert_token_classifier_ner_bc2gm_gene`
- `bert_token_classifier_ner_bc4chemd_chemicals`
- `bert_token_classifier_ner_bc5cdr_chemicals`
- `bert_token_classifier_ner_bc5cdr_disease`
- `bert_token_classifier_ner_jnlpba_cellular`
- `bert_token_classifier_ner_linnaeus_species`
- `bert_token_classifier_ner_ncbi_disease`
- `bert_token_classifier_ner_species`
- `bert_sequence_classifier_ade_augmented`
- `bert_sequence_classifier_health_mandates_stance_tweet`
- `bert_sequence_classifier_health_mandates_premise_tweet`
- `bert_sequence_classifier_treatement_changes_sentiment_tweet`
- `bert_sequence_classifier_drug_reviews_webmd`
- `bert_sequence_classifier_self_reported_age_tweet`
- `bert_sequence_classifier_self_reported_symptoms_tweet` => es
- `bert_sequence_classifier_self_reported_vaccine_status_tweet`
- `bert_sequence_classifier_self_reported_partner_violence_tweet`
- `bert_sequence_classifier_exact_age_reddit`
- `bert_sequence_classifier_self_reported_stress_tweet`
- `bert_token_classifier_disease_mentions_tweet` => es
- `bert_token_classifier_ner_ade_tweet_binary`
- `bert_token_classifier_ner_pathogen`
- `clinical_deidentification`
- `clinical_deidentification_slim`
- `umls_clinical_drugs_mapper`
- `umls_clinical_findings_mapper`
- `umls_disease_syndrome_mapper`
- `umls_major_concepts_mapper`
- `umls_drug_substance_mapper`
- `umls_drug_resolver_pipeline`
- `umls_clinical_findings_resolver_pipeline`
- `umls_disease_syndrome_resolver_pipeline`
- `umls_major_concepts_resolver_pipeline`
- `umls_drug_substance_resolver_pipeline`
- `classifierdl_health_mentions`
- `bert_sequence_classifier_health_mentions`
- `ner_medication_pipeline`
- `bert_sequence_classifier_vaccine_sentiment`
- `classifierdl_vaccine_sentiment`
- `bert_sequence_classifier_stressor`
- `re_ade_conversational`
- `medication_resolver_pipeline`


<div class="prev_ver h3-box" markdown="1">

## Versions

</div>

<ul class="pagination">
    <li>
        <a href="release_notes_4_0_0">Version 4.0.0</a>
    </li>
    <li>
        <strong>Version 4.0.2</strong>
    </li>
    <li>
        <a href="release_notes_4_1_0">Version 4.1.0</a>
    </li>
</ul>

<ul class="pagination owl-carousel pagination_big">
    <li><a href="release_notes_4_2_0">4.2.0</a></li>
    <li><a href="release_notes_4_1_0">4.1.0</a></li>
    <li class="active"><a href="release_notes_4_0_2">4.0.2</a></li>
    <li><a href="release_notes_4_0_0">4.0.0</a></li>
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