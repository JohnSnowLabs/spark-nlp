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

## 4.1.0

#### Highlights

+ Zero-Shot NER model to extract entities with no training dataset
+ 7 new clinical NER models in Spanish
+ 8 new clinical classification models in English and German related to public health topics (depression, covid sentiment, health mentions)
+ New pretrained chunk mapper model (`drug_ade_mapper`) to map drugs with their corresponding adverse drug events
+ A new pretrained resolver pipeline (`medication_resolver_pipeline`) to extract medications and resolve their adverse reactions (ADE), RxNorm, UMLS, NDC, SNOMED CT codes and action/treatments in clinical text with a single line of code.
+ Updated NER profiling pretrained pipelines with new NER models to allow running 64 clinical NER models at once
+ Core improvements and bug fixes
+ New and updated notebooks
+ 20+ new clinical models and pipelines added & updated in total

#### Zero-Shot NER model to Extract Entities With No Training Dataset

We are releasing the first of its kind Zero-Shot NER model that can detect any named entities without using any annotated dataset to train a model. It allows extracting entities by crafting appropriate prompts to query **any RoBERTa Question Answering model**.

See [Models Hub Page](https://nlp.johnsnowlabs.com/2022/08/29/zero_shot_ner_roberta_en.html) for more details.

*Example* :

```python
...
zero_shot_ner = ZeroShotNerModel.pretrained("zero_shot_ner_roberta", "en", "clincial/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("zero_shot_ner")\
    .setEntityDefinitions(
        {
            "PROBLEM": ["What is the disease?", "What is his symptom?", "What is her disease?", "What is his disease?",
                        "What is the problem?" ,"What does a patient suffer", 'What was the reason that the patient is admitted to the clinic?'],
            "DRUG": ["Which drug?", "Which is the drug?", "What is the drug?", "Which drug does he use?", "Which drug does she use?", "Which drug do I use?", "Which drug is prescribed for a symptom?"],
            "ADMISSION_DATE": ["When did patient admitted to a clinic?"],
            "PATIENT_AGE": ["How old is the patient?",'What is the age of the patient?']
        })\
...

sample_text = ["The doctor pescribed Majezik for my severe headache.",
               "The patient was admitted to the hospital for his colon cancer.",
               "27 years old patient was admitted to clinic on Sep 1st by Dr. X for a right-sided pleural effusion for thoracentesis."]
```

*Results* :

```bash
+------------------------------------------------+--------------+----------+
|                                           chunk|     ner_label|confidence|
+------------------------------------------------+--------------+----------+
|                                         Majezik|          DRUG|0.64671576|
|                                 severe headache|       PROBLEM| 0.5526346|
|                                    colon cancer|       PROBLEM| 0.8898498|
|                                    27 years old|   PATIENT_AGE| 0.6943085|
|                                         Sep 1st|ADMISSION_DATE|0.95646095|
|a right-sided pleural effusion for thoracentesis|       PROBLEM|0.50026613|
+------------------------------------------------+--------------+----------+
```





#### 7 New Clinical NER Models in Spanish

+ We are releasing 4 new `MedicalNerModel` and 3 new `MedicalBertForTokenClassifier` NER models in Spanish.

| model name                                          	| description                                                                	| predicted entities          	|
|-----------------------------------------------------	|----------------------------------------------------------------------------	|-----------------------------	|
| [ner_negation_uncertainty](https://nlp.johnsnowlabs.com/2022/08/13/ner_negation_uncertainty_es_3_0.html)                            	| This model detects relevant entities from Spanish medical texts            	| `NEG` `UNC` `USCO` `NSCO`   	|
| [disease_mentions_tweet](https://nlp.johnsnowlabs.com/2022/08/14/disease_mentions_tweet_es_3_0.html)                              	| This model detects disease mentions in Spanish tweets                      	| `ENFERMEDAD`                	|
| [ner_clinical_trials_abstracts](https://nlp.johnsnowlabs.com/2022/08/12/ner_clinical_trials_abstracts_es_3_0.html)                       	| This model detects relevant entities from Spanish clinical trial abstracts 	| `CHEM` `DISO` `PROC`        	|
| [ner_pharmacology](https://nlp.johnsnowlabs.com/2022/08/13/ner_pharmacology_es_3_0.html)                                    	| This model detects pharmacological entities from Spanish medical texts     	| `PROTEINAS` `NORMALIZABLES` 	|
| [bert_token_classifier_ner_clinical_trials_abstracts](https://nlp.johnsnowlabs.com/2022/08/11/bert_token_classifier_ner_clinical_trials_abstracts_es_3_0.html) 	| This model detects relevant entities from Spanish clinical trial abstracts 	| `CHEM` `DISO` `PROC`        	|
| [bert_token_classifier_negation_uncertainty](https://nlp.johnsnowlabs.com/2022/08/11/bert_token_classifier_negation_uncertainty_es_3_0.html)          	| This model detects relevant entities from Spanish medical texts            	| `NEG` `NSCO` `UNC` `USCO`   	|
| [bert_token_classifier_pharmacology](https://nlp.johnsnowlabs.com/2022/08/11/bert_token_classifier_pharmacology_es_3_0.html)                  	| This model detects pharmacological entities from Spanish medical texts     	| `PROTEINAS` `NORMALIZABLES` 	|


*Example* :

```python
...
ner = MedicalNerModel.pretrained('ner_clinical_trials_abstracts', "es", "clinical/models") \
	.setInputCols(["sentence", "token", "embeddings"]) \
	.setOutputCol("ner")

example_text=  """"Efecto de la suplementación con ácido fólico sobre los niveles de homocisteína total en pacientes en hemodiálisis. La hiperhomocisteinemia es un marcador de riesgo independiente de morbimortalidad cardiovascular. Hemos prospectivamente reducir los niveles de homocisteína total (tHcy) mediante suplemento con ácido fólico y vitamina B6 (pp), valorando su posible correlación con dosis de diálisis, función  residual y parámetros nutricionales.""""

```

*Results* :

```bash
+-----------------------------+---------+
|chunk                        |ner_label|
+-----------------------------+---------+
|suplementación               |PROC     |
|ácido fólico                 |CHEM     |
|niveles de homocisteína      |PROC     |
|hemodiálisis                 |PROC     |
|hiperhomocisteinemia         |DISO     |
|niveles de homocisteína total|PROC     |
|tHcy                         |PROC     |
|ácido fólico                 |CHEM     |
|vitamina B6                  |CHEM     |
|pp                           |CHEM     |
|diálisis                     |PROC     |
|función  residual            |PROC     |
+-----------------------------+---------+
```


#### 8 New Clinical Classification Models in English and German Related to Public Health Topics (Depression, Covid Sentiment, Health Mentions)

+ We are releasing 8 new `MedicalBertForSequenceClassification` models to classify text from social media data in English and German related to public health topics (depression, covid sentiment, health mentions)

| model name                                           	| description                                                                                                                                        	| predicted entities                          	|
|------------------------------------------------------	|----------------------------------------------------------------------------------------------------------------------------------------------------	|---------------------------------------------	|
| [bert_sequence_classifier_depression_binary](https://nlp.johnsnowlabs.com/2022/08/10/bert_sequence_classifier_depression_binary_en_3_0.html)           	| This model classifies whether a social media text expresses depression or not.                                                                     	| `no-depression` `depression`                	|
| [bert_sequence_classifier_health_mentions_gbert_large](https://nlp.johnsnowlabs.com/2022/08/10/bert_sequence_classifier_health_mentions_gbert_large_de_3_0.html) 	| This GBERT-large based model classifies public health mentions in German social media text.                                                        	| `non-health` `health-related`               	|
| [bert_sequence_classifier_health_mentions_medbert](https://nlp.johnsnowlabs.com/2022/08/10/bert_sequence_classifier_health_mentions_medbert_de_3_0.html)     	| This German-MedBERT based model classifies public health mentions in German social media text.                                                     	| `non-health` `health-related`               	|
| [bert_sequence_classifier_health_mentions_gbert](https://nlp.johnsnowlabs.com/2022/08/10/bert_sequence_classifier_health_mentions_gbert_de_3_0.html)       	| This GBERT-large based model classifies public health mentions in German social media text.                                                        	| `non-health` `health-related`               	|
| [bert_sequence_classifier_health_mentions_bert](https://nlp.johnsnowlabs.com/2022/08/10/bert_sequence_classifier_health_mentions_bert_de_3_0.html)        	| This bert-base-german based model classifies public health mentions in German social media text.                                                   	| `non-health` `health-related`               	|
| [bert_sequence_classifier_depression_twitter](https://nlp.johnsnowlabs.com/2022/08/09/bert_sequence_classifier_depression_twitter_en_3_0.html)          	| This PHS-BERT based model classifies whether tweets contain depressive text or not.                                                                	| `depression` `no-depression`                	|
| [bert_sequence_classifier_depression](https://nlp.johnsnowlabs.com/2022/08/09/bert_sequence_classifier_depression_en_3_0.html)                  	| This PHS-BERT based model classifies depression level of social media text into three levels.                                                      	| `no-depression` `minimum` `high-depression` 	|
| [bert_sequence_classifier_covid_sentiment](https://nlp.johnsnowlabs.com/2022/08/01/bert_sequence_classifier_covid_sentiment_en_3_0.html)             	| This BioBERT based sentiment analysis model classifies whether a tweet contains positive, negative, or neutral sentiments about COVID-19 pandemic. 	| `neutral` `positive` `negative`             	|

*Example* :

```python
...
sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_depression_twitter", "en", "clinical/models")\
     .setInputCols(["document","token"])\
     .setOutputCol("class")

example_text = ["Do what makes you happy, be with who makes you smile, laugh as much as you breathe, and love as long as you live!",
                "Everything is a lie, everyone is fake, I'm so tired of living"]
```


*Results* :

```bash
+------------------------------------------------------------------------------------------------------------------+---------------+
 |text                                                                                                             |result         |
 +-----------------------------------------------------------------------------------------------------------------+---------------+
 |Do what makes you happy, be with who makes you smile, laugh as much as you breathe, and love as long as you live!|[no-depression]|
 |Everything is a lie, everyone is fake, I am so tired of living.                                                  |[depression]   |
 +-----------------------------------------------------------------------------------------------------------------+---------------+
```

#### New Pretrained Chunk Mapper Model (`drug_ade_mapper`) to Map Drugs With Their Corresponding Adverse Drug Events

We are releasing new `drug_ade_mapper` pretrained chunk mapper model to map drugs with their corresponding adverse drug events.

See [Models Hub Page](https://nlp.johnsnowlabs.com/2022/08/23/drug_ade_mapper_en.html) for more details.

*Example* :

```python
...
chunkMapper = ChunkMapperModel.pretrained("drug_ade_mapper", "en", "clinical/models")\
      .setInputCols(["ner_chunk"])\
      .setOutputCol("mappings")\
      .setRels(["ADE"])
...

sample_text = "The patient was prescribed 1000 mg fish oil and multivitamins. She was discharged on zopiclone and ambrisentan."
```

*Results* :

```bash
+----------------+------------+-------------------------------------------------------------------------------------------+
|ner_chunk       |ade_mappings|all_relations                                                                              |
+----------------+------------+-------------------------------------------------------------------------------------------+
|1000 mg fish oil|Dizziness   |Myocardial infarction:::Nausea                                                             |
|multivitamins   |Erythema    |Acne:::Dry skin:::Skin burning sensation:::Inappropriate schedule of product administration|
|zopiclone       |Vomiting    |Malaise:::Drug interaction:::Asthenia:::Hyponatraemia                                      |
|ambrisentan     |Dyspnoea    |Therapy interrupted:::Death:::Dizziness:::Drug ineffective                                 |
+----------------+------------+-------------------------------------------------------------------------------------------+
```


#### A New Pretrained Resolver Pipeline (`medication_resolver_pipeline`) to Extract Medications and Resolve Their Adverse Reactions (ADE), RxNorm, UMLS, NDC, SNOMED CT Codes and Action/Treatments in Clinical Text.

We are releasing the `medication_resolver_pipeline` pretrained pipeline to extract medications and resolve their adverse reactions (ADE), RxNorm, UMLS, NDC, SNOMED CT codes and action/treatments in clinical text with a single line of code.

Also, you can use `medication_resolver_transform_pipeline` to use transform method of Spark.

See [Models Hub Page](https://nlp.johnsnowlabs.com/2022/09/01/medication_resolver_pipeline_en.html) for more details.


*Example* :

```python
from sparknlp.pretrained import PretrainedPipeline

sample_text = """The patient was prescribed Amlodopine Vallarta 10-320mg, Eviplera.
                 The other patient is given Lescol 40 MG and Everolimus 1.5 mg tablet."""

med_pipeline = PretrainedPipeline("medication_resolver_pipeline", "en", "clinical/models")
med_pipeline.annotate(sample_text)

med_transform_pipeline = PretrainedPipeline("medication_resolver_transform_pipeline", "en", "clinical/models")
med_transform_pipeline.transform(spark.createDataFrame([[sample_text]]).toDF("text"))
```

*Results* :

```bash
| chunk                        | ner_label   | ADE                         |   RxNorm | Action                     | Treatment                                  | UMLS     | SNOMED_CT   | NDC_Product   | NDC_Package   |
|:-----------------------------|:------------|:----------------------------|---------:|:---------------------------|:-------------------------------------------|:---------|:------------|:--------------|:--------------|
| Amlodopine Vallarta 10-320mg | DRUG        | Gynaecomastia               |   722131 | NONE                       | NONE                                       | C1949334 | 425838008   | 00093-7693    | 00093-7693-56 |
| Eviplera                     | DRUG        | Anxiety                     |   217010 | Inhibitory Bone Resorption | Osteoporosis                               | C0720318 | NONE        | NONE          | NONE          |
| Lescol 40 MG                 | DRUG        | NONE                        |   103919 | Hypocholesterolemic        | Heterozygous Familial Hypercholesterolemia | C0353573 | NONE        | 00078-0234    | 00078-0234-05 |
| Everolimus 1.5 mg tablet     | DRUG        | Acute myocardial infarction |  2056895 | NONE                       | NONE                                       | C4723581 | NONE        | 00054-0604    | 00054-0604-21 |
```

#### Updated NER Profiling Pretrained Pipelines With New NER Models to Allow Running 64 Clinical NER Models at Once

We have upadated `ner_profiling_clinical` and `ner_profiling_biobert` pretrained pipelines with the new NER models. When you run these pipelines over your text, now you will end up with the predictions coming out of **64 clinical NER models in `ner_profiling_clinical`** and **22 clinical NER models in `ner_profiling_biobert`** results.

You can check [ner_profiling_clinical](https://nlp.johnsnowlabs.com/2022/08/30/ner_profiling_clinical_en.html) and [ner_profiling_biobert](https://nlp.johnsnowlabs.com/2022/08/28/ner_profiling_biobert_en.html) Models Hub pages for more details and the NER model lists that these pipelines include.


#### Core Improvements and Bug Fixes

+ Updated HCC module (`from sparknlp_jsl.functions import profile`) with the new changes in HCC score calculation functions.
+ `AnnotationToolJsonReader`, `NerDLMetrics` and `StructuredDeidentification`: These annotators can be used on Spark 3.0 now.
+ `NerDLMetrics`:
  - Added `case_sensitive` parameter and case sensitivity issue in tokens is solved.
  - Added `drop_o` parameter to `computeMetricsFromDF` method and `dropO` parameter in `NerDLMetrics` class is **deprecated**.
+ `MedicalNerModel`: Inconsistent NER model results between different versions issue is solved.
+ `AssertionDLModel`: Unindexed chunks will be ignored by the `AssertionDLModel` instead of raising an exception.
+ `ContextualParserApproach`: These two issues are solved when using `ruleScope: "document"` configuration:
  - Wrong index computations of chunks after matching sub-tokens.
  - Including sub-token matches even though `completeMatchRegex: "true"`.


#### New and Updated Notebooks

+ We have a new [Zero-Shot Clinical NER Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.6.ZeroShot_Clinical_NER.ipynb) to show how to use zero-shot NER model.
+ We have updated [Medicare Risk Adjustment Score Calculation Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.1.Calculate_Medicare_Risk_Adjustment_Score.ipynb) with the new changes in HCC score calculation functions.
+ We have updated these notebooks with the new updates in NER profiling pretrained pipelines:
  - [Clinical Named Entity Recognition Model Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb)
  - [Pretrained Clinical Pipelines Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/11.Pretrained_Clinical_Pipelines.ipynb)
  - [Pretrained NER Profiling Pipelines Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/11.2.Pretrained_NER_Profiling_Pipelines.ipynb)
+ We have updated [Clinical Assertion Model Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/2.Clinical_Assertion_Model.ipynb) according to the bug fix in the training section.
+ We moved all Azure/AWS/Databricks notebooks to `products` folder in [spark-nlp-worksop](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/products) repo.

#### 20+ New Clinical Models and Pipelines Added & Updated in Total

+ `zero_shot_ner_roberta`
+ `medication_resolver_pipeline`
+ `medication_resolver_transform_pipeline`
+ `ner_profiling_clinical`
+ `ner_profiling_biobert`
+ `drug_ade_mapper`
+ `ner_negation_uncertainty`
+ `disease_mentions_tweet`
+ `ner_clinical_trials_abstracts`
+ `ner_pharmacology`
+ `bert_token_classifier_ner_clinical_trials_abstracts`
+ `bert_token_classifier_negation_uncertainty`
+ `bert_token_classifier_pharmacology`
+ `bert_sequence_classifier_depression_binary`
+ `bert_sequence_classifier_health_mentions_gbert_large`
+ `bert_sequence_classifier_health_mentions_medbert`
+ `bert_sequence_classifier_health_mentions_gbert`
+ `bert_sequence_classifier_health_mentions_bert`
+ `bert_sequence_classifier_depression_twitter`
+ `bert_sequence_classifier_depression`
+ `bert_sequence_classifier_covid_sentiment`





<div class="prev_ver h3-box" markdown="1">

## Previous versions

</div>
<ul class="pagination">
    <li>
        <a href="spark_nlp_healthcare_versions/release_notes_4_0_2">Versions 4.0.2</a>
    </li>
    <li>
        <strong>Versions 4.1.0</strong>
    </li>
</ul>
<ul class="pagination owl-carousel pagination_big">
    <li class="active"><a href="/licensed_release_notes">4.1.0</a></li>
    <li><a href="spark_nlp_healthcare_versions/release_notes_4_0_2">4.0.2</a></li>
    <li><a href="spark_nlp_healthcare_versions/release_notes_4_0_0">4.0.0</a></li>
    <li><a href="spark_nlp_healthcare_versions/release_notes_3_5_3">3.5.3</a></li>
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
