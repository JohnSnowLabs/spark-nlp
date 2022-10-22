---
layout: docs
header: true
seotitle: Spark NLP for Healthcare | John Snow Labs
title: Spark NLP for Healthcare Release Notes 3.4.1
permalink: /docs/en/spark_nlp_healthcare_versions/release_notes_3_4_1
key: docs-licensed-release-notes
modify_date: 2021-07-14
show_nav: true
sidebar:
    nav: sparknlp-healthcare
---

## 3.4.1

We are glad to announce that Spark NLP Healthcare 3.4.1 has been released!

#### Highlights

+ Brand new Spanish deidentification NER models
+ Brand new Spanish deidentification pretrained pipeline
+ New clinical NER model to detect supplements
+ New RxNorm sentence entity resolver model
+ New `EntityChunkEmbeddings` annotator
+ New `MedicalBertForSequenceClassification` annotator
+ New `MedicalDistilBertForSequenceClassification` annotator
+ New `MedicalDistilBertForSequenceClassification` and `MedicalBertForSequenceClassification` models
+ Redesign of the `ContextualParserApproach` annotator
+ `getClasses` method in `RelationExtractionModel` and `RelationExtractionDLModel` annotators
+ Label customization feature for `RelationExtractionModel` and `RelationExtractionDL` models
+ `useBestModel` parameter in `MedicalNerApproach` annotator
+ Early stopping feature in `MedicalNerApproach` annotator
+ Multi-Language support for faker and regex lists of `Deidentification` annotator
+ Spark 3.2.0 compatibility for the entire library
+ Saving visualization feature in `spark-nlp-display` library
+ Deploying a custom Spark NLP image (for opensource, healthcare, and Spark OCR) to an enterprise version of Kubernetes: OpenShift
+ New speed benchmarks table on databricks
+ New & Updated Notebooks
+ List of recently updated or added models

#### Brand New Spanish Deidentification NER Models

We trained two new NER models to find PHI data (protected health information) that may need to be deidentified in **Spanish**. `ner_deid_generic` and `ner_deid_subentity` models are trained with in-house annotations. Both also are available for using Roberta Spanish Clinical Embeddings and sciwiki 300d.

+ `ner_deid_generic` : Detects 7 PHI entities in Spanish (`DATE`, `NAME`, `LOCATION`, `PROFESSION`, `CONTACT`, `AGE`, `ID`).

+ `ner_deid_subentity` : Detects 13 PHI sub-entities in Spanish (`PATIENT`, `HOSPITAL`, `DATE`, `ORGANIZATION`, `E-MAIL`, `USERNAME`, `LOCATION`, `ZIP`, `MEDICALRECORD`, `PROFESSION`, `PHONE`, `DOCTOR`, `AGE`).

*Example* :

```bash
...
embeddings = WordEmbeddingsModel.pretrained("embeddings_sciwiki_300d","es","clinical/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")

deid_ner = MedicalNerModel.pretrained("ner_deid_generic", "es", "clinical/models")\
    .setInputCols(["sentence", "token", "embeddings"])\
    .setOutputCol("ner")

deid_sub_entity_ner = MedicalNerModel.pretrained("ner_deid_subentity", "es", "clinical/models")\
    .setInputCols(["sentence", "token", "embeddings"])\
    .setOutputCol("ner_sub_entity")
...

text = """Antonio Pérez Juan, nacido en Cadiz, España. Aún no estaba vacunado, se infectó con Covid-19 el dia 14/03/2020
y tuvo que ir al Hospital. Fue tratado con anticuerpos monoclonales en la Clinica San Carlos.."""
result = model.transform(spark.createDataFrame([[text]], ["text"]))
```

*Results* :

```bash
| chunk              | ner_deid_generic_chunk | ner_deid_subentity_chunk |
|--------------------|------------------------|--------------------------|
| Antonio Pérez Juan | NAME                   | PATIENT                  |
| Cádiz              | LOCATION               | LOCATION                 |
| España             | LOCATION               | LOCATION                 |
| 14/03/2022         | DATE                   | DATE                     |
| Clínica San Carlos | LOCATION               | HOSPITAL                 |
```

#### Brand New Spanish Deidentification Pretrained Pipeline

We developed a clinical deidentification pretrained pipeline that can be used to deidentify PHI information from **Spanish** medical texts. The PHI information will be masked and obfuscated in the resulting text. The pipeline can mask, fake or obfuscate the following entities: `AGE`, `DATE`, `PROFESSION`, `E-MAIL`, `USERNAME`, `LOCATION`, `DOCTOR`, `HOSPITAL`, `PATIENT`, `URL`, `IP`, `MEDICALRECORD`, `IDNUM`, `ORGANIZATION`, `PHONE`, `ZIP`, `ACCOUNT`, `SSN`, `PLATE`, `SEX` and `IPADDR`.

```bash
from sparknlp.pretrained import PretrainedPipeline
deid_pipeline = PretrainedPipeline("clinical_deidentification", "es", "clinical/models")

sample_text = """Datos del paciente. Nombre:  Jose . Apellidos: Aranda Martinez. NHC: 2748903. NASS: 26 37482910."""

result = deid_pipe.annotate(text)

print("\n".join(result['masked']))
print("\n".join(result['masked_with_chars']))
print("\n".join(result['masked_fixed_length_chars']))
print("\n".join(result['obfuscated']))
```
*Results*:

```bash
Masked with entity labels
------------------------------
Datos del paciente. Nombre:  <PATIENT> . Apellidos: <PATIENT>. NHC: <SSN>. NASS: <SSN> <SSN>

Masked with chars
------------------------------
Datos del paciente. Nombre:  [**] . Apellidos: [*************]. NHC: [*****]. NASS: [**] [******]

Masked with fixed length chars
------------------------------
Datos del paciente. Nombre:  **** . Apellidos: ****. NHC: ****. NASS: **** ****

Obfuscated
------------------------------
Datos del paciente. Nombre:  Sr. Lerma . Apellidos: Aristides Gonzalez Gelabert. NHC: BBBBBBBBQR648597. NASS: 041010000011 RZRM020101906017 04.
```

#### New Clinical NER Model to Detect Supplements

We are releasing `ner_supplement_clinical` model that can extract benefits of using drugs for certain conditions. It can label detected entities as `CONDITION` and `BENEFIT`. Also this model is trained on the dataset that is released by Spacy in their HealthSea product. Here is the benchmark comparison of both versions:

|Entity|Spark NLP| Spacy-HealthSea|
|-|-|-|
|BENEFIT|0.8729641|0.8330684|
|CONDITION|0.8339274|0.8333333|

*Example* :

```bash
...
clinical_ner = MedicalNerModel.pretrained("ner_supplement_clinical", "en", "clinical/models") \
      .setInputCols(["sentence", "token", "embeddings"]) \
      .setOutputCol("ner_tags")
...

results = ner_model.transform(spark.createDataFrame([["Excellent!. The state of health improves, nervousness disappears, and night sleep improves. It also promotes hair and nail growth."]], ["text"]))
```

*Results* :

```bash
+------------------------+---------------+
| chunk                  | ner_label     |
+------------------------+---------------+
| nervousness            | CONDITION     |
| night sleep improves   | BENEFIT       |
| hair                   | BENEFIT       |
| nail                   | BENEFIT       |
+------------------------+---------------+
```

#### New RxNorm Sentence Entity Resolver Model

`sbiobertresolve_rxnorm_augmented_re` : This model maps clinical entities and concepts (like drugs/ingredients) to RxNorm codes without specifying the relations between the entities (relations are calculated on the fly inside the annotator) using sbiobert_base_cased_mli Sentence Bert Embeddings (EntityChunkEmbeddings).

*Example* :

```python
...
rxnorm_resolver = SentenceEntityResolverModel\
      .pretrained("sbiobertresolve_rxnorm_augmented_re", "en", "clinical/models")\
      .setInputCols(["entity_chunk_embeddings"])\
      .setOutputCol("rxnorm_code")\
      .setDistanceFunction("EUCLIDEAN")
...
```

#### New `EntityChunkEmbeddings` Annotator

We have a new `EntityChunkEmbeddings` annotator to compute a weighted average vector representing entity-related vectors. The model's input usually consists of chunks of recognized named entities produced by MedicalNerModel. We can specify relations between the entities by the `setTargetEntities()` parameter, and the internal Relation Extraction model finds related entities and creates a chunk. Embedding for the chunk is calculated according to the weights specified in the `setEntityWeights()` parameter.

For instance, the chunk `warfarin sodium 5 MG Oral Tablet` has `DRUG`, `STRENGTH`, `ROUTE`, and `FORM` entity types. Since DRUG label is the most prominent label for resolver models, now we can assign weight to prioritize DRUG label (i.e `{"DRUG": 0.8, "STRENGTH": 0.2, "ROUTE": 0.2, "FORM": 0.2}` as shown below). In other words, embeddings of these labels are multipled by the assigned weights such as `DRUG` by `0.8`.

For more details and examples, please check [Sentence Entity Resolvers with EntityChunkEmbeddings Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.2.Sentence_Entity_Resolvers_with_EntityChunkEmbeddings.ipynb) in the Spark NLP workshop repo.

*Example* :

```python
...

drug_chunk_embeddings = EntityChunkEmbeddings()\
    .pretrained("sbiobert_base_cased_mli","en","clinical/models")\
    .setInputCols(["ner_chunks", "dependencies"])\
    .setOutputCol("drug_chunk_embeddings")\
    .setMaxSyntacticDistance(3)\
    .setTargetEntities({"DRUG": ["STRENGTH", "ROUTE", "FORM"]})\
    .setEntityWeights({"DRUG": 0.8, "STRENGTH": 0.2, "ROUTE": 0.2, "FORM": 0.2})

rxnorm_resolver = SentenceEntityResolverModel\
    .pretrained("sbiobertresolve_rxnorm_augmented_re", "en", "clinical/models")\
    .setInputCols(["drug_chunk_embeddings"])\
    .setOutputCol("rxnorm_code")\
    .setDistanceFunction("EUCLIDEAN")

rxnorm_weighted_pipeline_re = Pipeline(
    stages = [
        documenter,
        sentence_detector,
        tokenizer,
        embeddings,
        posology_ner_model,
        ner_converter,
        pos_tager,
        dependency_parser,
        drug_chunk_embeddings,
        rxnorm_resolver])

sampleText = ["The patient was given metformin 500 mg, 2.5 mg of coumadin and then ibuprofen.",
              "The patient was given metformin 400 mg, coumadin 5 mg, coumadin, amlodipine 10 MG"]

data_df = spark.createDataFrame(sample_df)
results = rxnorm_weighted_pipeline_re.fit(data_df).transform(data_df)
```

The internal relation extraction creates the chunks here, and the embedding is computed according to the weights.

*Results* :
```bash
+-----+----------------+--------------------------+--------------------------------------------------+
|index|           chunk|rxnorm_code_weighted_08_re|                                      Concept_Name|
+-----+----------------+--------------------------+--------------------------------------------------+
|    0|metformin 500 mg|                    860974|metformin hydrochloride 500 MG:::metformin 500 ...|
|    0| 2.5 mg coumadin|                    855313|warfarin sodium 2.5 MG [Coumadin]:::warfarin so...|
|    0|       ibuprofen|                   1747293|ibuprofen Injection:::ibuprofen Pill:::ibuprofe...|
|    1|metformin 400 mg|                    332809|metformin 400 MG:::metformin 250 MG Oral Tablet...|
|    1|   coumadin 5 mg|                    855333|warfarin sodium 5 MG [Coumadin]:::warfarin sodi...|
|    1|        coumadin|                    202421|Coumadin:::warfarin sodium 2 MG/ML Injectable S...|
|    1|amlodipine 10 MG|                    308135|amlodipine 10 MG Oral Tablet:::amlodipine 10 MG...|
+-----+----------------+--------------------------+--------------------------------------------------+
```

#### New `MedicalBertForSequenceClassification` Annotator

We developed a new annotator called `MedicalBertForSequenceClassification`. It can load BERT Models with sequence classification/regression head on top (a linear layer on top of the pooled output) e.g. for multi-class document classification tasks.

#### New `MedicalDistilBertForSequenceClassification` Annotator

We developed a new annotator called `MedicalDistilBertForSequenceClassification`. It can load DistilBERT Models with sequence classification/regression head on top (a linear layer on top of the pooled output) e.g. for multi-class document classification tasks.

#### New `MedicalDistilBertForSequenceClassification` and `MedicalBertForSequenceClassification` Models

We are releasing a new `MedicalDistilBertForSequenceClassification` model and three new `MedicalBertForSequenceClassification` models.

- `bert_sequence_classifier_ade_biobert`: a classifier for detecting if a sentence is talking about a possible ADE (`TRUE`, `FALSE`)

- `bert_sequence_classifier_gender_biobert`: a classifier for detecting the gender of the main subject of the sentence (`MALE`, `FEMALE`, `UNKNOWN`)

- `bert_sequence_classifier_pico_biobert`: a classifier for detecting the class of a sentence according to PICO framework (`CONCLUSIONS`, `DESIGN_SETTING`,`INTERVENTION`, `PARTICIPANTS`, `FINDINGS`, `MEASUREMENTS`, `AIMS`)

*Example* :

```python
...
sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_pico", "en", "clinical/models")\
    .setInputCols(["document","token"])\
    .setOutputCol("class")
...

sample_text = "To compare the results of recording enamel opacities using the TF and modified DDE indices."

result = sequence_clf_model.transform(spark.createDataFrame([[sample_text]]).toDF("text"))
```

*Results* :

```
+-------------------------------------------------------------------------------------------+-----+
|text                                                                                       |label|
+-------------------------------------------------------------------------------------------+-----+
|To compare the results of recording enamel opacities using the TF and modified DDE indices.|AIMS |
+-------------------------------------------------------------------------------------------+-----+
```


+ `distilbert_sequence_classifier_ade` : This model is a DistilBertForSequenceClassification model for classifying clinical texts whether they contain ADE (`TRUE`, `FALSE`).

*Example* :

```python
...
sequenceClassifier = MedicalDistilBertForSequenceClassification\
      .pretrained('distilbert_sequence_classifier_ade', 'en', 'clinical/models') \
      .setInputCols(['token', 'document']) \
      .setOutputCol('class')
...

sample_text = "I felt a bit drowsy and had blurred vision after taking Aspirin."

result = sequence_clf_model.transform(spark.createDataFrame([[sample_text]]).toDF("text"))
```

*Results* :

```
+----------------------------------------------------------------+-----+
|text                                                            |label|
+----------------------------------------------------------------+-----+
|I felt a bit drowsy and had blurred vision after taking Aspirin.| True|
+----------------------------------------------------------------+-----+
```


#### Redesign of the `ContextualParserApproach` Annotator

- We've dropped the annotator's `contextMatch` parameter and removed the need for a `context` field when feeding a JSON configuration file to the annotator. Context information can now be fully defined using the `prefix`, `suffix` and `contextLength` fields in the JSON configuration file.
- We've also fixed issues with the `contextException` field in the JSON configuration file - it was mismatching values in documents with several sentences and ignoring exceptions situated to the right of a word/token.
- The `ruleScope` field in the JSON configuration file can now be set to `document` instead of `sentence`. This allows you to match multi-word entities like "New York" or "Salt Lake City". You can do this by setting `"ruleScope" : "document"` in the JSON configuration file and feeding a dictionary (csv or tsv) to the annotator with its `setDictionary` parameter. These changes also mean that we've dropped the `updateTokenizer` parameter since the new capabilities of `ruleScope` improve the user experience for matching multi-word entities.
- You can now feed in a dictionary in your chosen format - either vertical or horizontal. You can set that with the following parameter: `setDictionary("dictionary.csv", options={"orientation":"vertical"})`
- Lastly, there was an improvement made to the confidence value calculation process to better measure successful hits.

For more explanation and examples, please check this [Contextual Parser medium article](https://medium.com/spark-nlp/contextual-parser-increased-flexibility-extracting-entities-in-spark-nlp-123ed58672f0) and [Contextual Parser Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.2.Contextual_Parser_Rule_Based_NER.ipynb).

#### `getClasses` Method in `RelationExtractionModel` and `RelationExtractionDLModel` Annotators

Now you can use `getClasses()` method for checking the relation labels of RE models (RelationExtractionModel and RelationExtractionDLModel) like MedicalNerModel().

*Example* :
```python
clinical_re_Model = RelationExtractionModel()\
    .pretrained("re_temporal_events_clinical", "en", 'clinical/models')\
    .setInputCols(["embeddings", "pos_tags", "ner_chunks", "dependencies"])\
    .setOutputCol("relations")\

clinical_re_Model.getClasses()
```

*Output* :
```output
['OVERLAP', 'BEFORE', 'AFTER']
```

####  Label Customization Feature for `RelationExtractionModel` and `RelationExtractionDL` Models

We are releasing label customization feature for Relation Extraction and Relation Extraction DL models by using `.setCustomLabels()` parameter.

*Example* :

```python
...
reModel = RelationExtractionModel.pretrained("re_ade_clinical", "en", 'clinical/models')\
    .setInputCols(["embeddings", "pos_tags", "ner_chunks", "dependencies"])\
    .setOutputCol("relations")\
    .setMaxSyntacticDistance(10)\
    .setRelationPairs(["drug-ade, ade-drug"])\
    .setCustomLabels({"1": "is_related", "0": "not_related"})

redl_model = RelationExtractionDLModel.pretrained('redl_ade_biobert', 'en', "clinical/models") \
    .setPredictionThreshold(0.5)\
    .setInputCols(["re_ner_chunks", "sentences"]) \
    .setOutputCol("relations")\
    .setCustomLabels({"1": "is_related", "0": "not_related"})
...

sample_text = "I experienced fatigue and muscle cramps after taking Lipitor but no more adverse after passing Zocor."
result = model.transform(spark.createDataFrame([[sample_text]]).toDF('text'))
```

*Results* :

```
+-----------+-------+-------------+-------+-------+----------+
|   relation|entity1|       chunk1|entity2| chunk2|confidence|
+-----------+-------+-------------+-------+-------+----------+
| is_related|    ADE|      fatigue|   DRUG|Lipitor| 0.9999825|
|not_related|    ADE|      fatigue|   DRUG|  Zocor| 0.9960077|
| is_related|    ADE|muscle cramps|   DRUG|Lipitor|       1.0|
|not_related|    ADE|muscle cramps|   DRUG|  Zocor|   0.94971|
+-----------+-------+-------------+-------+-------+----------+
```


#### `useBestModel` Parameter in `MedicalNerApproach` Annotator

Introducing `useBestModel` param in MedicalNerApproach annotator. This param preserves and restores the model that has achieved the best performance at the end of the training. The priority is metrics from testDataset (micro F1), metrics from validationSplit (micro F1), and if none is set it will keep track of loss during the training.

*Example* :
```python
med_ner = MedicalNerApproach()\
    .setInputCols(["sentence", "token", "embeddings"])\
    .setLabelColumn("label")\
    .setOutputCol("ner")\
    ...
    ...
    .setUseBestModel(True)\
```

#### Early Stopping Feature in `MedicalNerApproach` Annotator

Introducing `earlyStopping` feature for MedicalNerApproach(). You can stop training at the point when the perforfmance on test/validation dataset starts to degrage. Two params are added to MedicalNerApproach() in order to use this feature:

+ `earlyStoppingCriterion` : (float) This is used set the minimal improvement of the test metric to terminate training. The metric monitored is the same as the metrics used in `useBestModel` (macro F1 when using test/validation set, loss otherwise). Default is 0 which means no early stopping is applied.

+ `earlyStoppingPatience`: (int), the number of epoch without improvement which will be tolerated. Default is 0, which means that early stopping will occur at the first time when performance in the current epoch is no better than in the previous epoch.

*Example* :

```python
med_ner = MedicalNerApproach()\
    .setInputCols(["sentence", "token", "embeddings"])\
    .setLabelColumn("label")\
    .setOutputCol("ner")\
    ...
    ...
    .setTestDataset(test_data_parquet_path)\
    .setEarlyStoppingCriterion(0.01)\
    .setEarlyStoppingPatience(3)\
```

#### Multi-Language Support for Faker and Regex Lists of `Deidentification` Annotator

We have a new `.setLanguage()` parameter in order to use internal Faker and Regex list for multi-language texts. When you are working with German and Spanish texts for a Deidentification, you can set this parameter to `de` for German and `es` for Spanish. Default value of this parameter is `en`.

*Example* :

```python
deid_obfuscated = DeIdentification()\
      .setInputCols(["sentence", "token", "ner_chunk"]) \
      .setOutputCol("obfuscated") \
      .setMode("obfuscate")\
      .setLanguage('de')\
      .setObfuscateRefSource("faker")\
```

#### Spark 3.2.0 Compatibility for the Entire Library

Now we can use the [Spark 3.2.0](https://spark.apache.org/docs/3.2.0/) version for Spark NLP for Healthcare by setting `spark32=True` in `sparknlp_jsl.start()` function.

```bash
! pip install --ignore-installed -q pyspark==3.2.0
```

```bash
import sparknlp_jsl

spark = sparknlp_jsl.start(SECRET, spark32=True)
```

#### Saving Visualization Feature in `spark-nlp-display` Library

We have a new `save_path` parameter in `spark-nlp-display` library for saving any visualization results in Spark NLP.

*Example* :

```bash
from sparknlp_display import NerVisualizer

visualiser = NerVisualizer()

visualiser.display(light_result[0], label_col='ner_chunk', document_col='document', save_path="display_result.html")
```

#### Deploying a Custom Spark NLP Image (for opensource, healthcare, and Spark OCR) to an Enterprise Version of Kubernetes: OpenShift

Spark NLP for opensource, healthcare, and SPARK OCR is now available for Openshift - enterprise version of Kubernetes. For deployment, please refer to:

Github Link: https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/platforms/openshift

Youtube: https://www.youtube.com/watch?v=FBes-6ylFrM&ab_channel=JohnSnowLabs

#### New Speed Benchmarks Table on Databricks

We prepared a speed benchmark table by running a clinical BERT For Token Classification model pipeline on various number of repartitioning and writing the results to parquet or delta formats. You can find the details here : [Clinical Bert For Token Classification Benchmark Experiment](https://nlp.johnsnowlabs.com/docs/en/benchmark#clinical-bert-for-token-classification-benchmark-experiment).

#### New & Updated Notebooks

+ We have updated our existing workshop notebooks with v3.4.0 by adding new features and functionalities.
+ You can find the workshop notebooks updated with previous versions in the branches named with the relevant version.
+ We have updated the [ContextualParser Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.2.Contextual_Parser_Rule_Based_NER.ipynb) with the new updates in this version.
+ We have a new [Sentence Entity Resolvers with EntityChunkEmbeddings Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.2.Sentence_Entity_Resolvers_with_EntityChunkEmbeddings.ipynb) for the new `EntityChunkEmbeddings` annotator.

**To see more, please check : [Spark NLP Healthcare Workshop Repo](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings/Healthcare)**

#### List of Recently Updated or Added Models

- `bert_sequence_classifier_ade_en`
- `bert_sequence_classifier_gender_biobert_en`
- `bert_sequence_classifier_pico_biobert_en`
- `distilbert_sequence_classifier_ade_en`
- `bert_token_classifier_ner_supplement_en`
- `deid_pipeline_es`
- `ner_deid_generic_es`
- `ner_deid_generic_roberta_es`
- `ner_deid_subentity_es`
- `ner_deid_subentity_roberta_es`
- `ner_nature_nero_clinical_en`
- `ner_supplement_clinical_en`
- `sbiobertresolve_clinical_abbreviation_acronym_en`
- `sbiobertresolve_rxnorm_augmented_re`

**For all Spark NLP for healthcare models, please check : [Models Hub Page](https://nlp.johnsnowlabs.com/models?edition=Spark+NLP+for+Healthcare)**

<div class="prev_ver h3-box" markdown="1">

## Versions

</div>

<ul class="pagination">
    <li>
        <a href="release_notes_3_4_0">Version 3.4.0</a>
    </li>
    <li>
        <strong>Version 3.4.1</strong>
    </li>
    <li>
        <a href="release_notes_3_4_2">Version 3.4.2</a>
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
    <li class="active"><a href="release_notes_3_4_1">3.4.1</a></li>
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