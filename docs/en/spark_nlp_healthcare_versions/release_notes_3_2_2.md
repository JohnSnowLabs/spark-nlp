---
layout: docs
header: true
seotitle: Spark NLP for Healthcare | John Snow Labs
title: Spark NLP for Healthcare Release Notes 3.2.2
permalink: /docs/en/spark_nlp_healthcare_versions/release_notes_3_2_2
key: docs-licensed-release-notes
modify_date: 2021-07-14
show_nav: true
sidebar:
    nav: sparknlp-healthcare
---

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

<div class="prev_ver h3-box" markdown="1">

## Versions

</div>

<ul class="pagination">
    <li>
        <a href="release_notes_3_2_1">Version 3.2.1</a>
    </li>
    <li>
        <strong>Version 3.2.2</strong>
    </li>
    <li>
        <a href="release_notes_3_2_3">Version 3.2.3</a>
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
    <li class="active"><a href="release_notes_3_2_2">3.2.2</a></li>
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