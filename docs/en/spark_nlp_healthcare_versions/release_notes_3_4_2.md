---
layout: docs
header: true
seotitle: Spark NLP for Healthcare | John Snow Labs
title: Spark NLP for Healthcare Release Notes 3.4.2
permalink: /docs/en/spark_nlp_healthcare_versions/release_notes_3_4_2
key: docs-licensed-release-notes
modify_date: 2021-07-14
show_nav: true
sidebar:
    nav: sparknlp-healthcare
---

## 3.4.2
We are glad to announce that Spark NLP Healthcare 3.4.2 has been released!

#### Highlights

 + New RCT Classifier, NER models and pipeline (Deidentification)
 + Setting the scope window (target area) dynamically in Assertion Status detection models
 + Reading JSON files (exported from ALAB) from HDFS with `AnnotationJsonReader`
 + Allow users to write Tensorflow graphs to HDFS
 + Serving Spark NLP on APIs
 + Updated documentation on installing Spark NLP for Healthcare in AWS EMR (Jupyter, Livy, Yarn, Hadoop)
 + New series of notebooks to reproduce the academic papers published by our colleagues
 + PySpark tutorial notebooks to let non-Spark users get started with Apache Spark ecosystem in Python
 + New & updated notebooks
 + List of recently updated or added models

#### New RCT Classifier, NER Models and Pipeline (Deidentification)

We are releasing a new `bert_sequence_classifier_rct_biobert` model, four new Spanish deidentification NER models (`ner_deid_generic_augmented`, `ner_deid_subentity_augmented`, `ner_deid_generic_roberta_augmented`, `ner_deid_subentity_roberta_augmented`) and a pipeline (`clinical_deidentification_augmented`).

 + `bert_sequence_classifier_rct_biobert`: This model can classify the sections within abstract of scientific articles regarding randomized clinical trials (RCT) (`BACKGROUND`, `CONCLUSIONS`, `METHODS`, `OBJECTIVE`, `RESULTS`).

*Example* :

```python
...
sequenceClassifier_model = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_rct_biobert", "en", "clinical/models")\
  .setInputCols(["document",'token'])\
  .setOutputCol("class")
...

sample_text = "Previous attempts to prevent all the unwanted postoperative responses to major surgery with an epidural hydrophilic opioid , morphine , have not succeeded . The authors ' hypothesis was that the lipophilic opioid fentanyl , infused epidurally close to the spinal-cord opioid receptors corresponding to the dermatome of the surgical incision , gives equal pain relief but attenuates postoperative hormonal and metabolic responses more effectively than does systemic fentanyl ."

result = sequence_clf_model.transform(spark.createDataFrame([[sample_text]]).toDF("text"))

>> class: 'BACKGROUND'
```


+ `ner_deid_generic_augmented`, `ner_deid_subentity_augmented`, `ner_deid_generic_roberta_augmented`, `ner_deid_subentity_roberta_augmented` models and `clinical_deidentification_augmented` pipeline : You can use either `sciwi-embeddings` (300 dimensions) or the Roberta Clinical Embeddings (infix `_roberta_`) with these NER models. These models and pipeline are different to their non-augmented versions in the following:

  - They are trained with more data, now including an in-house annotated deidentification dataset;
  - New `SEX` tag is available for all of them. This tag is now included in the NER and has been improved with more rules in the ContextualParsers of the pipeline, resulting in having a bigger recall to detect the sex of the patient.
  - New `STREET`, `CITY` and `COUNTRY` entities are added to subentity versions.

For more details and examples, please check [Clinical Deidentification in Spanish notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/4.2.Clinical_Deidentification_in_Spanish.ipynb).

*Example* :

```python
...
embeddings = WordEmbeddingsModel.pretrained("embeddings_sciwiki_300d","es","clinical/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")

deid_ner = MedicalNerModel.pretrained("ner_deid_generic_augmented", "es", "clinical/models")\
    .setInputCols(["sentence", "token", "embeddings"])\
    .setOutputCol("ner")

deid_sub_entity_ner = MedicalNerModel.pretrained("ner_deid_subentity_augmented", "es", "clinical/models")\
    .setInputCols(["sentence", "token", "embeddings"])\
    .setOutputCol("ner_sub_entity")
...
```

*Results* :
```bash
chunk                    entity_subentity    entity_generic
-----------------------  ------------------  ----------------
Antonio Miguel Martínez  PATIENT             NAME
un varón                 SEX                 SEX
35                       AGE                 AGE
auxiliar de enfermería   PROFESSION          PROFESSION
Cadiz                    CITY                LOCATION
España                   COUNTRY             LOCATION
Clinica San Carlos       HOSPITAL            LOCATION
```

#### Setting the Scope Window (Target Area) Dynamically in Assertion Status Detection Models

This parameter allows you to train the Assertion Status Models to focus on specific context windows when resolving the status of a NER chunk. The window is in format `[X,Y]` being `X` the number of tokens to consider on the left of the chunk, and `Y` the max number of tokens to consider on the right. Let's take a look at what different windows mean:

- By default, the window is `[-1,-1]` which means that the Assertion Status will look at all of the tokens in the sentence/document (up to a maximum of tokens set in `setMaxSentLen()`).
- `[0,0]` means "don't pay attention to any token except the ner_chunk", what basically is not considering any context for the Assertion resolution.
- `[9,15]` is what empirically seems to be the best baseline, meaning that we look up to 9 tokens on the left and 15 on the right of the ner chunk to understand the context and resolve the status.

Check this [scope window tuning assertion status detection notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/2.1.Scope_window_tuning_assertion_status_detection.ipynb) that illustrates the effect of the different windows and how to properly **fine-tune** your AssertionDLModels to get the best of them.

*Example* :

```python
assertion_status = AssertionDLApproach() \
          .setGraphFolder("assertion_dl/") \
          .setInputCols("sentence", "chunk", "embeddings") \
          .setOutputCol("assertion") \
          ...
          ...
          .setScopeWindow([9, 15])     # NEW! Scope Window!
```

#### Reading JSON Files (Exported from ALAB) From HDFS with `AnnotationJsonReader`

Now we can read the dataframe from a HDFS that we read the files from in our cluster.

*Example* :

```python
filename = "hdfs:///user/livy/import.json"
reader = AnnotationToolJsonReader(assertion_labels = ['AsPresent', 'AsAbsent', 'AsConditional', 'AsHypothetical', 'Family', 'AsPossible', 'AsElse'])
df = reader.readDataset(spark, filename)
```

#### Allow Users Write Tensorflow Graphs to HDFS

Now we can save custom Tensorflow graphs to the HDFS that mainly being used in a cluster environment.

```python
tf_graph.build("ner_dl", build_params={"embeddings_dim": 200, "nchars": 128, "ntags": 12, "is_medical": 1}, model_location="hdfs:///user/livy", model_filename="auto")
```

#### Serving Spark NLP on APIs

Two new notebooks and a series of blog posts / Medium articles have been created to guide Spark NLP users to serve Spark NLP on a RestAPI.

* The notebooks can be found [here](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/RestAPI).
* The articles can be found in the Technical Documentation of Spark NLP, available [here](https://nlp.johnsnowlabs.com/docs/en/quickstart) and also in Medium:
	* [Serving Spark NLP via API (1/3): Microsoft’s Synapse ML](https://medium.com/@jjmcarrascosa/serving-spark-nlp-via-api-1-3-microsoft-synapse-ml-2c77a3f61f9d)
	* [Serving Spark NLP via API (2/3): FastAPI and LightPipelines](https://medium.com/@jjmcarrascosa/serving-spark-nlp-via-api-2-3-fastapi-and-lightpipelines-218d1980c9fc)
	* [Serving Spark NLP via API (3/3): Databricks Jobs and MLFlow Serve APIs](https://medium.com/@jjmcarrascosa/serving-spark-nlp-via-api-3-3-databricks-and-mlflow-serve-apis-4ef113e7fac4)

The difference between both approaches are the following:
+ `SynapseML` is a Microsoft Azure Open Source library used to carry out ML at scale. In this case, we use the Spark Serving feature, that leverages Spark Streaming and adds a web server with a Load Balancer, allowing concurrent processing of Spark NLP calls. Best approach if you look for scalability with Load Balancing.
+ `FastAPI` + `LightPipelines`: A solution to run Spark NLP using a FastAPI webserver. It uses LightPipelines, what means having a very good performance but not leveraging Spark Clusters. Also, no Load Balancer is available in the suggestion, but you can create your own. Best approach if you look for performance.
+ `Databricks` and `MLFlow`: Using MLFlow Serve or Databricks Jobs APIs to serve for inference Spark NLP pipelines from within Databricks. Best approach if you look for scalability within Databricks.


#### Updated Documentation on Installing Spark NLP For Healthcare in AWS EMR (Jupyter, Livy, Yarn, Hadoop)

Ready-to-go Spark NLP for Healthcare environment in AWS EMR. Full instructions are [here](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/platforms/emr).

#### New Series of Notebooks to Reproduce the Academic Papers Published by Our Colleagues

You can find all these notebooks [here](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/academic)

#### PySpark Tutorial Notebooks to Let Non-Spark Users to Get Started with Apache Spark Ecosystem in Python

John Snow Labs has created a series of 8 notebooks to go over PySpark from zero to hero. Notebooks cover PySpark essentials, DataFrame creation, querying, importing data from different formats, functions / udfs, Spark MLLib examples (regression, classification, clustering) and Spark NLP best practises (usage of parquet, repartition, coalesce, custom annotators, etc).

You can find all these notebooks [here](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/PySpark).

#### New & Updated Notebooks

+ `Series of academic notebooks` : A new series of academic paper notebooks, available [here](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/academic)
+ `Clinical_Deidentification_in_Spanish.ipynb`: A notebook showcasing Clinical Deidentification in Spanish, available [here](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/4.2.Clinical_Deidentification_in_Spanish.ipynb).
+ `Clinical_Deidentification_Comparison.ipynb`: A new series of comparisons between different Deidentification libraries. So far, it contains Spark NLP for Healthcare and ScrubaDub with Spacy Transformers. Available [here](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/4.3.Clinical_Deidentification_Comparison.ipynb).
+ `Scope_window_tuning_assertion_status_detection.ipynb`: How to finetune Assertion Status using the Scope Window. Available [here](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/2.1.Scope_window_tuning_assertion_status_detection.ipynb)
+ `Clinical_Longformer_vs_BertSentence_&_USE.ipynb`: A Comparison of how Clinical Longformer embeddings, averaged by the Sentence Embeddings annotator, performs compared to BioBert and UniversalSentenceEncoding. Link [here](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/clinical_text_classification/3.Clinical_Longformer_vs_BertSentence_%26_USE.ipynb).
+ `Serving_SparkNLP_with_Synapse.ipynb`: Serving SparkNLP for production purposes using Synapse ML. Available [here](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/RestAPI/Serving_SparkNLP_with_Synapse.ipynb)
+ `Serving_SparkNLP_with_FastAPI_and_LP.ipynb`: Serving SparkNLP for production purposes using FastAPI, RestAPI and LightPipelines. Available [here](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/RestAPI/Serving_SparkNLP_with_FastAPI_and_LP.ipynb)
+ `Series of PySpark tutorial notebooks`: Available [here](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/PySpark)

#### List of Recently Updated or Added Models

+ `sbiobertresolve_hcpcs`
+ `bert_sequence_classifier_rct_biobert`
+ `ner_deid_generic_augmented_es`
+ `ner_deid_subentity_augmented_es`
+ `ner_deid_generic_roberta_augmented_es`
+ `ner_deid_subentity_roberta_augmented_es`
+ `clinical_deidentification_augmented_es`

**For all Spark NLP for healthcare models, please check : [Models Hub Page](https://nlp.johnsnowlabs.com/models?edition=Spark+NLP+for+Healthcare)**

<div class="prev_ver h3-box" markdown="1">

## Versions

</div>

<ul class="pagination">
    <li>
        <a href="release_notes_3_4_1">Version 3.4.1</a>
    </li>
    <li>
        <strong>Version 3.4.2</strong>
    </li>
    <li>
        <a href="release_notes_3_5_0">Version 3.5.0</a>
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
    <li class="active"><a href="release_notes_3_4_2">3.4.2</a></li>
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