---
layout: docs
header: true
seotitle: Spark NLP for Healthcare | John Snow Labs
title: Spark NLP for Healthcare Release Notes 3.1.1
permalink: /docs/en/spark_nlp_healthcare_versions/release_notes_3_1_1
key: docs-licensed-release-notes
modify_date: 2021-07-14
show_nav: true
sidebar:
    nav: sparknlp-healthcare
---

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

<div class="prev_ver h3-box" markdown="1">

## Versions

</div>

<ul class="pagination">
    <li>
        <a href="release_notes_3_1_0">Version 3.1.0</a>
    </li>
    <li>
        <strong>Version 3.1.1</strong>
    </li>
    <li>
        <a href="release_notes_3_1_2">Version 3.1.2</a>
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
    <li><a href="release_notes_3_3_0">3.3.0</a></li>
    <li><a href="release_notes_3_2_3">3.2.3</a></li>
    <li><a href="release_notes_3_2_2">3.2.2</a></li>
    <li><a href="release_notes_3_2_1">3.2.1</a></li>
    <li><a href="release_notes_3_2_0">3.2.0</a></li>
    <li><a href="release_notes_3_1_3">3.1.3</a></li>
    <li><a href="release_notes_3_1_2">3.1.2</a></li>
    <li class="active"><a href="release_notes_3_1_1">3.1.1</a></li>
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