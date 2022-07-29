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
ner_model = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_exact_age_reddit", "en", "clinical/models")\
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


## 3.5.3

#### Highlights

+ New `rxnorm_mapper` model
+ New `ChunkMapperFilterer` annotator to filter `ChunkMapperModel` results
+ New features
  - Add the `setReplaceLabels` parameter that allows replacing the non-conventional labels without using an external source file in the `NerConverterInternal()`.
  - Case sensitivity can be set in `ChunkMapperApproach` and `ChunkMapperModel` through `setLowerCase()` parameter.
  - Return multiple relations at a time in `ChunkMapperModel` models via `setRels()` parameter.
  - Filter the multi-token chunks separated with whitespace in `ChunkMapperApproach` by `setAllowMultiTokenChunk()` parameter.
+ New license validation policy in License Validator.
+ Bug fixes
+ Updated notebooks
+ List of recently updated or added models

#### New `rxnorm_mapper` Model

We are releasing `rxnorm_mapper` model that maps clinical entities and concepts to corresponding rxnorm codes.

See [Model Hub Page](https://nlp.johnsnowlabs.com/2022/06/07/rxnorm_mapper_en_3_0.html) for details.

*Example* :
```python
...
chunkerMapper = ChunkMapperModel.pretrained("rxnorm_mapper", "en", "clinical/models")\
       .setInputCols(["ner_chunk"])\
       .setOutputCol("mappings")\
       .setRel("rxnorm_code")
...

sample_text = "The patient was given Zyrtec 10 MG, Adapin 10 MG Oral Capsule, Septi-Soothe 0.5 Topical Spray"
```

*Results* :

```bash
 +------------------------------+---------------+
 |chunk                         |rxnorm_mappings|
 +------------------------------+---------------+
 |Zyrtec 10 MG                  |1011483        |
 |Adapin 10 MG Oral Capsule     |1000050        |
 |Septi-Soothe 0.5 Topical Spray|1000046        |
 +------------------------------+---------------+
```

#### New `ChunkMapperFilterer` Annotator to Filter `ChunkMapperModel` Results
`ChunkMapperFilterer` annotator allows filtering of the chunks that were passed through the `ChunkMapperModel`.
If `setReturnCriteria()` is set as `"success"`, only the chunks which are mapped by `ChunkMapperModel` are returned. Otherwise, if `setReturnCriteria()` is set as `"fail"`, only the chunks which are not mapped by ChunkMapperModel are returned.

*Example* :
```python
...
cfModel = ChunkMapperFilterer() \
            .setInputCols(["ner_chunk","mappings"]) \
            .setOutputCol("chunks_filtered")\
            .setReturnCriteria("success") #or "fail"
...
sample_text = "The patient was given Warfarina Lusa and amlodipine 10 mg. Also, he was given Aspagin, coumadin 5 mg and metformin"

```



`.setReturnCriteria("success")` *Results* :

```bash
+-----+---+--------------+--------------+
|begin|end|        entity|      mappings|
+-----+---+--------------+--------------+
|   22| 35|          DRUG|Warfarina Lusa|
+-----+---+--------------+--------------+
```

`.setReturnCriteria("fail")` *Results* :

```bash
+-----+---+--------+------------+
|begin|end|  entity|  not mapped|
+-----+---+--------+------------+
|   41| 50|    DRUG|  amlodipine|
|   80| 86|    DRUG|     Aspagin|
|   89| 96|    DRUG|    coumadin|
|  115|123|    DRUG|   metformin|
+-----+---+--------+------------+
```



#### New Features:

##### Add `setReplaceLabels` Parameter That Allows Replacing the Non-Conventional Labels Without Using an External Source File in the `NerConverterInternal()`.

Now you can replace the labels in NER models with custom labels by using `.setReplaceLabels` parameter with `NerConverterInternal` annotator. In this way, you will not need to use any other external source file to replace the labels with custom ones.

*Example* :

```python
...
clinical_ner = MedicalNerModel.pretrained("ner_jsl", "en", "clinical/models")\
    .setInputCols(["sentence","token", "word_embeddings"])\
    .setOutputCol("ner")

ner_converter_original = NerConverterInternal()\
    .setInputCols(["sentence", "token", "ner"]) \
    .setOutputCol("original_label")

ner_converter_replaced = NerConverterInternal()\
    .setInputCols(["sentence", "token", "ner"]) \
    .setOutputCol("replaced_label")\
    .setReplaceLabels({"Drug_Ingredient" : "Drug",'Drug_BrandName':'Drug'})
...

sample_text = "The patient was given Warfarina Lusa and amlodipine 10 mg. Also, he was given Aspagin, coumadin 5 mg, and metformin"
```

*Results* :

```bash
+--------------+-----+---+---------------+--------------+
|chunk         |begin|end|original_label |replaced_label|
+--------------+-----+---+---------------+--------------+
|Warfarina Lusa|22   |35 |Drug_BrandName |Drug          |
|amlodipine    |41   |50 |Drug_Ingredient|Drug          |
|10 mg         |52   |56 |Strength       |Strength      |
|he            |65   |66 |Gender         |Gender        |
|Aspagin       |78   |84 |Drug_BrandName |Drug          |
|coumadin      |87   |94 |Drug_Ingredient|Drug          |
|5 mg          |96   |99 |Strength       |Strength      |
|metformin     |106  |114|Drug_Ingredient|Drug          |
+--------------+-----+---+---------------+--------------+
```


##### Case Sensitivity in `ChunkMapperApproach` and `ChunkMapperModel` Through `setLowerCase()` Parameter

The case status of `ChunkMapperApproach` and `ChunkMapperModel` can be set by using `setLowerCase()` parameter.

*Example* :

```python
...
chunkerMapperapproach = ChunkMapperApproach() \
        .setInputCols(["ner_chunk"]) \
        .setOutputCol("mappings") \
        .setDictionary("mappings.json") \
        .setRel("action") \
        .setLowerCase(True) #or False

...

sentences = [["""The patient was given Warfarina lusa and amlodipine 10 mg, coumadin 5 mg.
                 The patient was given Coumadin"""]]
```

`setLowerCase(True)` *Results* :

```bash
+------------------------+-----------+
|chunk                   |mapped     |
+------------------------+-----------+
|Warfarina lusa          |540228     |
|amlodipine              |329526     |
|coumadin                |202421     |
|Coumadin                |202421     |
+------------------------+-----------+

```

`setLowerCase(False)` *Results* :

```bash
+------------------------+-----------+
|chunk                   |mapped     |
+------------------------+-----------+
|Warfarina lusa          |NONE       |
|amlodipine              |329526     |
|coumadin                |NONE       |
|Coumadin                |202421     |
+------------------------+-----------+
```

##### Return Multiple Relations At a Time In ChunkMapper Models Via `setRels()` Parameter
Multiple relations for the same chunk can be set with the `setRels()` parameter in both `ChunkMapperApproach` and `ChunkMapperModel`.

*Example* :
```python
...
chunkerMapperapproach = ChunkMapperApproach() \
        .setInputCols(["ner_chunk"]) \
        .setOutputCol("mappings") \
        .setDictionary("mappings.json") \
        .setRels(["action","treatment"]) \
        .setLowerCase(True) \
...

sample_text = "The patient was given Warfarina Lusa."
```

*Results* :

```bash
+-----+---+--------------+-------------+---------+
|begin|end|        entity|     mappings| relation|
+-----+---+--------------+-------------+---------+
|   22| 35|Warfarina Lusa|Anticoagulant|   action|
|   22| 35|Warfarina Lusa|Heart Disease|treatment|
+-----+---+--------------+-------------+---------+
```

##### Filter the Multi-Token Chunks Separated With Whitespace in `ChunkMapperApproach` and `ChunkMapperModel` by `setAllowMultiTokenChunk()` Parameter

The chunks that include multi-tokens separated by a whitespace, can be filtered by using `setAllowMultiTokenChunk()` parameter.

*Example* :
```python
...
chunkerMapper = ChunkMapperApproach() \
        .setInputCols(["ner_chunk"]) \
        .setOutputCol("mappings") \
        .setDictionary("mappings.json") \
        .setLowerCase(True) \
        .setRels(["action", "treatment"]) \
        .setAllowMultiTokenChunk(False)
...

sample_text = "The patient was given Warfarina Lusa"
```

`setAllowMultiTokenChunk(False)` *Results* :

```bash
+-----+---+--------------+--------+--------+
|begin|end|         chunk|mappings|relation|
+-----+---+--------------+--------+--------+
|   22| 35|Warfarina Lusa|    NONE|    null|
+-----+---+--------------+--------+--------+
```

`setAllowMultiTokenChunk(True)` *Results* :

```bash
+-----+---+--------------+-------------+---------+
|begin|end|         chunk|     mappings| relation|
+-----+---+--------------+-------------+---------+
|   22| 35|Warfarina Lusa|Anticoagulant|   action|
|   22| 35|Warfarina Lusa|Heart Disease|treatment|
+-----+---+--------------+-------------+---------+
```



#### New License Validation Policies in License Validator

A new version of the License Validator has been included in Spark NLP for Healthcare. This License Validator checks the compatibility between the type of your license and the environment you are using, allowing the license to be used only for the environment it was requested (single-node, cluster, databricks, etc) and the number of concurrent sessions (floating or not-floating). You can check which type of license you have in [my.johnsnowlabs.com](https://my.johnsnowlabs.com/) -> My Subscriptions.

If your license stopped working, please contact [support@johnsnowlabs.com](mailto:support@johnsnowlabs.com) so that it can be checked the difference between the environment your license was requested for and the one it's currently being used.

#### Bug Fixes

We fixed some issues in `AnnotationToolJsonReader` tool, `DrugNormalizer` and `ContextualParserApproach` annotators.

+ **`DrugNormalizer`** : Fixed some issues that affect the performance.
+ **`ContextualParserApproach`** : Fixed the issue in the computation of indices for documents with more than one sentence while defining the rule-scope field as a document.
+ **`AnnotationToolJsonReader`** : Fixed an issue where relation labels were not being extracted from the Annotation Lab json file export.

#### Updated Notebooks

+ [Clinical Named Entity Recognition Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb) <br/>
 `.setReplaceLabels` parameter example was added.
+ [Chunk Mapping Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/26.Chunk_Mapping.ipynb) <br/>
 New case sensitivity, selecting multiple relations, filtering multi-token chunks and `ChunkMapperFilterer` features were added.

#### List of Recently Updated Models

- `sbiobertresolve_icdo_augmented`
- `rxnorm_mapper`

For all Spark NLP for healthcare models, please check: [Models Hub Page](https://nlp.johnsnowlabs.com/models?edition=Spark+NLP+for+Healthcare)


<div class="prev_ver h3-box" markdown="1">

## Previos versions

</div>
<ul class="pagination">
    <li>
        <a href="spark_nlp_healthcare_versions/release_notes_3_5_2">Versions 3.5.2</a>
    </li>
    <li>
        <strong>Versions 3.5.3</strong>
    </li>
</ul>
<ul class="pagination owl-carousel pagination_big">
    <li class="active"><a href="spark_nlp_healthcare_versions/release_notes_3_5_3">3.5.3</a></li>
    <li><a href="spark_nlp_healthcare_versions/release_notes_3_5_2">3.5.2</a></li>
    <li><a href="spark_nlp_healthcare_versions/release_notes_3_5_1">3.5.1</a></li>
    <li><a href="spark_nlp_healthcare_versions/release_notes_3_5_0">3.5.0</a></li>
    <li><a href="spark_nlp_healthcare_versions/release_notes_3_4_2">3.4.2</a></li>
    <li><a href="spark_nlp_healthcare_versions/release_notes_3_4_1">3.4.1</a></li>
    <li><a href="spark_nlp_healthcare_versions/release_notes_3_4_0">3.4.0</a></li>
    <li><a href="spark_nlp_healthcare_versions/release_notes_3_3_4">3.3.4</a></li>
    <li><a href="spark_nlp_healthcare_versions/release_notes_3_3_2">3.3.2</a></li>
    <li><a href="spark_nlp_healthcare_versions/release_notes_3_3_1">3.3.1</a></li>
    <li><a href="spark_nlp_healthcare_versions/release_notes_3_3_0">3.3.0</a></li>
    <li><a href="spark_nlp_healthcare_versions/release_notes_3_2_3">3.2.3</a></li>
    <li><a href="spark_nlp_healthcare_versions/release_notes_3_2_2">3.2.2</a></li>
    <li><a href="spark_nlp_healthcare_versions/release_notes_3_2_1">3.2.1</a></li>
    <li><a href="spark_nlp_healthcare_versions/release_notes_3_2_0">3.2.0</a></li>
    <li><a href="spark_nlp_healthcare_versions/release_notes_3_1_3">3.1.3</a></li>
    <li><a href="spark_nlp_healthcare_versions/release_notes_3_1_2">3.1.2</a></li>
    <li><a href="spark_nlp_healthcare_versions/release_notes_3_1_1">3.1.1</a></li>
    <li><a href="spark_nlp_healthcare_versions/release_notes_3_1_0">3.1.0</a></li>
    <li><a href="spark_nlp_healthcare_versions/release_notes_3_0_3">3.0.3</a></li>
    <li><a href="spark_nlp_healthcare_versions/release_notes_3_0_2">3.0.2</a></li>
    <li><a href="spark_nlp_healthcare_versions/release_notes_3_0_1">3.0.1</a></li>
    <li><a href="spark_nlp_healthcare_versions/release_notes_3_0_0">3.0.0</a></li>
    <li><a href="spark_nlp_healthcare_versions/release_notes_2_7_6">2.7.6</a></li>
    <li><a href="spark_nlp_healthcare_versions/release_notes_2_7_5">2.7.5</a></li>
    <li><a href="spark_nlp_healthcare_versions/release_notes_2_7_4">2.7.4</a></li>
    <li><a href="spark_nlp_healthcare_versions/release_notes_2_7_3">2.7.3</a></li>
    <li><a href="spark_nlp_healthcare_versions/release_notes_2_7_2">2.7.2</a></li>
    <li><a href="spark_nlp_healthcare_versions/release_notes_2_7_1">2.7.1</a></li>
    <li><a href="spark_nlp_healthcare_versions/release_notes_2_7_0">2.7.0</a></li>
    <li><a href="spark_nlp_healthcare_versions/release_notes_2_6_2">2.6.2</a></li>
    <li><a href="spark_nlp_healthcare_versions/release_notes_2_6_0">2.6.0</a></li>
    <li><a href="spark_nlp_healthcare_versions/release_notes_2_5_5">2.5.5</a></li>
    <li><a href="spark_nlp_healthcare_versions/release_notes_2_5_3">2.5.3</a></li>
    <li><a href="spark_nlp_healthcare_versions/release_notes_2_5_2">2.5.2</a></li>
    <li><a href="spark_nlp_healthcare_versions/release_notes_2_5_0">2.5.0</a></li>
    <li><a href="spark_nlp_healthcare_versions/release_notes_2_4_6">2.4.6</a></li>
    <li><a href="spark_nlp_healthcare_versions/release_notes_2_4_5">2.4.5</a></li>
    <li><a href="spark_nlp_healthcare_versions/release_notes_2_4_2">2.4.2</a></li>
    <li><a href="spark_nlp_healthcare_versions/release_notes_2_4_1">2.4.1</a></li>
    <li><a href="spark_nlp_healthcare_versions/release_notes_2_4_0">2.4.0</a></li>
</ul>
