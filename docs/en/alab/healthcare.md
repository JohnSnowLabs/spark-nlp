---
layout: docs
comment: no
header: true
seotitle: Annotation Lab | John Snow Labs
title: Healthcare NLP Integration
permalink: /docs/en/alab/healthcare
key: docs-training
modify_date: "2022-11-19"
use_language_switcher: "Python-Scala"
show_nav: true
sidebar:
  nav: annotation-lab
---

<style>
.shell-output pre.highlight {
    background-color: #efefef !important;
    color: #4e4e4e;
}

.shell-output code {
  font-family: monospace;
}

pre {
  max-height: 500px;
}

.table_wrapper{
  display: block;
  overflow-x: auto;
  white-space: nowrap;
}

.article__content table {
  display: table;
  font-size: 12px;
}

.article__content table th,
.article__content table td {
  padding-top: 2px;
  padding-bottom: 2px;
  text-align: right;
}
</style>

Spark NLP for Healthcare now has an easy to use module for interacting with Annotation Lab with minimal code. In this section, you can find the instructions for performing specific operations using the annotation lab module of the Spark NLP for Healthcare library. You can execute these instructions in the Jupyter notebook or any other notebook of your preference.

Before trying out the instructions in the following sub-sections, you will need to perform some initial set up to configure the Spark NLP for Healthcare library and start a spark session.

```py
import json
import os

from google.colab import files

license_keys = files.upload()

with open(list(license_keys.keys())[0]) as f:
    license_keys = json.load(f)

# Defining license key-value pairs as local variables
locals().update(license_keys)

# Adding license key-value pairs to environment variables
os.environ.update(license_keys)
```

> **NOTE:** Upload widget is only available when the cell has been executed in the current browser session. Please rerun this cell to enable.

{:.shell-output}

```sh
Saving jsl_keys.json to jsl_keys (2).json
```

```py
# Installing pyspark and spark-nlp
! pip install --upgrade -q pyspark==3.1.2 spark-nlp==$PUBLIC_VERSION

# Installing Spark NLP Healthcare
! pip install --upgrade -q spark-nlp-jsl==$JSL_VERSION  --extra-index-url https://pypi.johnsnowlabs.com/$SECRET

# Installing Spark NLP Display Library for visualization
! pip install -q spark-nlp-display
```

{:.shell-output}

```sh
     |████████████████████████████████| 212.4 MB 51 kB/s
     |████████████████████████████████| 616 kB 56.5 MB/s
     |████████████████████████████████| 198 kB 52.8 MB/s
  Building wheel for pyspark (setup.py) ... done
     |████████████████████████████████| 206 kB 2.9 MB/s
     |████████████████████████████████| 95 kB 2.4 MB/s
     |████████████████████████████████| 66 kB 4.9 MB/s
     |████████████████████████████████| 1.6 MB 44.7 MB/s
```

```py
import pandas as pd
import requests
import json
from zipfile import ZipFile
from io import BytesIO
import os
from pyspark.ml import Pipeline,PipelineModel
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from sparknlp.annotator import *
from sparknlp_jsl.annotator import *
from sparknlp.base import *
import sparknlp_jsl
import sparknlp

import warnings
warnings.filterwarnings('ignore')

params = {"spark.driver.memory":"16G",
          "spark.kryoserializer.buffer.max":"2000M",
          "spark.driver.maxResultSize":"2000M"}

print("Spark NLP Version :", sparknlp.version())
print("Spark NLP_JSL Version :", sparknlp_jsl.version())

spark = sparknlp_jsl.start(license_keys['SECRET'],params=params)

spark
```

{:.shell-output}

```sh
Spark NLP Version : 4.1.0
Spark NLP_JSL Version : 4.1.0
SparkSession - in-memory

SparkContext

Spark UI

Version v3.1.2
Master local[*]
AppName Spark NLP Licensed
```

**Using already exported JSON to generate training data - _No Credentials Required_**

```py
# import the module
from sparknlp_jsl.alab import AnnotationLab
alab = AnnotationLab()
```

**Start with an already exported JSON.**

```sh
# downloading demo json
!wget https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp-workshop/master/tutorials/Annotation_Lab/data/alab_demo.json
```

{:.shell-output}

```sh
--2022-09-29 18:47:21--  https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp-workshop/master/tutorials/Annotation_Lab/data/alab_demo.json
Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 185.199.110.133, 185.199.111.133, 185.199.108.133, ...
Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|185.199.110.133|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 66538 (65K) [text/plain]
Saving to: ‘alab_demo.json’

alab_demo.json      100%[===================>]  64.98K  --.-KB/s    in 0.01s

2022-09-29 18:47:21 (5.43 MB/s) - ‘alab_demo.json’ saved [66538/66538]
```

## Generating training data for different models

{:.info}
No Alab Credentials required. Only exported JSON is required.

### Classification Model

Here, we will generate data for training a classification model.

```py
alab.get_classification_data(

    # required: path to Annotation Lab JSON export
    input_json_path='alab_demo.json',

    # optional: set to True to select ground truth completions, False to select latest completions,
    # defaults to False
    ground_truth=True

    )
```

{:.shell-output}

```sh
Processing 14 annotation(s).
```

**Output:**

|     | task_id | task_title | text                                              | class    |
| --- | ------- | ---------- | ------------------------------------------------- | -------- |
| 0   | 2       | Note 2     | The patient is a 5-month-old infant who presen... | [Female] |
| 1   | 3       | Note 3     | The patient is a 21-day-old male here for 2 da... | [Male]   |
| 2   | 1       | Note 1     | On 18/08 patient declares she has a headache s... | [Female] |

<br />

### NER Model

Here, we will convert the JSON export into a CoNLL format suitable for training an NER model.

```py
alab.get_conll_data(

    # required: Spark session with spark-nlp-jsl jar
    spark=spark,

    # required: path to Annotation Lab JSON export
    input_json_path="alab_demo.json",

    # required: name of the CoNLL file to save
    output_name="conll_demo",

    # optional: path for CoNLL file saving directory, defaults to 'exported_conll'
    save_dir="exported_conll",

    # optional: set to True to select ground truth completions, False to select latest completions,
    # defaults to False
    ground_truth=True,

    # optional: labels to exclude from CoNLL; these are all assertion labels and irrelevant NER labels,
    # defaults to empty list
    excluded_labels=['ABSENT'],

    # optional: set a pattern to use regex tokenizer, defaults to regular tokenizer if pattern not defined
    regex_pattern="\\s+|(?=[-.:;*+,$&%\\[\\]])|(?<=[-.:;*+,$&%\\[\\]])"

    # optional: list of Annotation Lab task IDs to exclude from CoNLL, defaults to empty list
    # excluded_task_ids = [2, 3]

    # optional: list of Annotation Lab task titles to exclude from CoNLL, defaults to None
    # excluded_task_titles = ['Note 1']

)
```

{:.shell-output}

```sh
sentence_detector_dl_healthcare download started this may take some time.
Approximate size to download 367.3 KB
[OK!]
pos_clinical download started this may take some time.
Approximate size to download 1.5 MB
[OK!]
Spark NLP LightPipeline is created
sentence_detector_dl_healthcare download started this may take some time.
Approximate size to download 367.3 KB
[OK!]
Spark NLP LightPipeline is created
Attempting to process: Task ID# 1
Task ID# 1 is included
Attempting to process: Task ID# 2
Task ID# 2 is included
Attempting to process: Task ID# 3
Task ID# 3 is included
Saved in location: exported_conll/conll_demo.conll

Printing first 30 lines of CoNLL for inspection:

['-DOCSTART- -X- -1- O\n\n',
 'On II II O\n',
 '18/08 MC MC B-DATE\n',
 'patient NN NN O\n',
 'declares NNS NNS O\n',
 'she PN PN O\n',
 'has VHZ VHZ O\n',
 'a DD DD O\n',
 'headache NN NN B-PROBLEM\n',
 'since CS CS O\n',
 '06/08 MC MC B-DATE\n',
 ', NN NN O\n',
 'needs VVZ VVZ O\n',
 'to TO TO O\n',
 'get VVI VVI O\n',
 'a DD DD O\n',
 'head NN NN B-TEST\n',
 'CT NN NN I-TEST\n',
 ', NN NN O\n',
 'and CC CC O\n',
 'appears VVZ VVZ O\n',
 'anxious JJ JJ B-PROBLEM\n',
 'when CS CS O\n',
 'she PN PN O\n',
 'walks RR RR O\n',
 'fast JJ JJ O\n',
 '. NN NN O\n',
 'No NN NN O\n',
 'alopecia NN NN B-PROBLEM\n',
 'noted VVNJ VVNJ O\n']
```

<br />

### Assertion Model

Here, we will convert the JSON export into a dataframe suitable for training an assertion model.

```py
alab.get_assertion_data(

    # required: SparkSession with spark-nlp-jsl jar
    spark=spark,

    # required: path to Annotation Lab JSON export
    input_json_path = 'alab_demo.json',

    # required: annotated assertion labels to train on
    assertion_labels = ['ABSENT'],

    # required: relevant NER labels that are assigned assertion labels
    relevant_ner_labels = ['PROBLEM', 'TREATMENT'],

    # optional: set to True to select ground truth completions, False to select latest completions,
    # defaults to False
    ground_truth = True,

    # optional: assertion label to assign to entities that have no assertion labels, defaults to None
    unannotated_label = 'PRESENT',

    # optional: set a pattern to use regex tokenizer, defaults to regular tokenizer if pattern not defined
    regex_pattern = "\\s+|(?=[-.:;*+,$&%\\[\\]])|(?<=[-.:;*+,$&%\\[\\]])",

    # optional: set the strategy to control the number of occurrences of the unannotated assertion label
    # in the output dataframe, options are 'weighted' or 'counts', 'weighted' allows to sample using a
    # fraction, 'counts' allows to sample using absolute counts, defaults to None
    unannotated_label_strategy = 'weighted',

    # optional: dictionary in the format {'ENTITY_LABEL': sample_weight_or_counts} to control the number of
    # occurrences of the unannotated assertion label in the output dataframe, where 'ENTITY_LABEL' are the
    # NER labels that are assigned the unannotated assertion label, and sample_weight_or_counts should be
    # between 0 and 1 if `unannotated_label_strategy` is 'weighted' or between 0 and the max number of
    # occurrences of that NER label if `unannotated_label_strategy` is 'counts'
    unannotated_label_strategy_dict = {'PROBLEM': 0.5, 'TREATMENT': 0.5},

    # optional: list of Annotation Lab task IDs to exclude from output dataframe, defaults to None
    # excluded_task_ids = [2, 3]

    # optional: list of Annotation Lab task titles to exclude from output dataframe, defaults to None
    # excluded_task_titles = ['Note 1']

)
```

{:.shell-output}

```sh
sentence_detector_dl_healthcare download started this may take some time.
Approximate size to download 367.3 KB
[OK!]
Spark NLP LightPipeline is created
Processing Task ID# 2
Processing Task ID# 3
Processing Task ID# 1
```

**Output:**

|     | task_id | title  | text                                              | target                      | ner_label | label   | start | end |
| --- | ------- | ------ | ------------------------------------------------- | --------------------------- | --------- | ------- | ----- | --- |
| 0   | 1       | Note 1 | On 18/08 patient declares she has a headache s... | headache                    | PROBLEM   | PRESENT | 7     | 7   |
| 1   | 1       | Note 1 | On 18/08 patient declares she has a headache s... | alopecia                    | PROBLEM   | ABSENT  | 27    | 27  |
| 2   | 1       | Note 1 | On 18/08 patient declares she has a headache s... | pain                        | PROBLEM   | ABSENT  | 32    | 32  |
| 3   | 2       | Note 2 | Mom states she had no fever.                      | fever                       | PROBLEM   | ABSENT  | 5     | 5   |
| 4   | 2       | Note 2 | She had no difficulty breathing and her cough ... | difficulty breathing        | PROBLEM   | ABSENT  | 3     | 4   |
| 5   | 2       | Note 2 | She had no difficulty breathing and her cough ... | cough                       | PROBLEM   | PRESENT | 7     | 7   |
| 6   | 2       | Note 2 | She had no difficulty breathing and her cough ... | dry                         | PROBLEM   | PRESENT | 11    | 11  |
| 7   | 2       | Note 2 | She had no difficulty breathing and her cough ... | hacky                       | PROBLEM   | PRESENT | 13    | 13  |
| 8   | 2       | Note 2 | At that time, physical exam showed no signs of... | flu                         | PROBLEM   | ABSENT  | 10    | 10  |
| 9   | 3       | Note 3 | The patient is a 21-day-old male here for 2 da... | congestion                  | PROBLEM   | PRESENT | 15    | 15  |
| 10  | 3       | Note 3 | The patient is a 21-day-old male here for 2 da... | suctioning yellow discharge | TREATMENT | PRESENT | 23    | 25  |
| 11  | 3       | Note 3 | The patient is a 21-day-old male here for 2 da... | perioral cyanosis           | PROBLEM   | ABSENT  | 47    | 48  |
| 12  | 3       | Note 3 | One day ago, mom also noticed a tactile temper... | tactile temperature         | PROBLEM   | PRESENT | 8     | 9   |

<br />

### Relation Extraction Model

Here, we will convert the JSON export into a dataframe suitable for training a relation extraction model.

```py
alab.get_relation_extraction_data(

    # required: Spark session with spark-nlp-jsl jar
    spark=spark,

    # required: path to Annotation Lab JSON export
    input_json_path='alab_demo.json',

    # optional: set to True to select ground truth completions, False to select latest completions,
    # defaults to False
    ground_truth=True,

    # optional: set to True to assign a relation label between entities where no relation was annotated,
    # defaults to False
    negative_relations=True,

    # optional: all assertion labels that were annotated in the Annotation Lab, defaults to None
    assertion_labels=['ABSENT'],

    # optional: plausible pairs of entities for relations, separated by a '-', use the same casing as the
    # annotations, include only one relation direction, defaults to all possible pairs of annotated entities
    relation_pairs=['DATE-PROBLEM','TREATMENT-PROBLEM','TEST-PROBLEM'],

    # optional: set the strategy to control the number of occurrences of the negative relation label
    # in the output dataframe, options are 'weighted' or 'counts', 'weighted' allows to sample using a
    # fraction, 'counts' allows to sample using absolute counts, defaults to None
    negative_relation_strategy='weighted',

    # optional: dictionary in the format {'ENTITY1-ENTITY2': sample_weight_or_counts} to control the number of
    # occurrences of negative relations in the output dataframe for each entity pair, where 'ENTITY1-ENTITY2'
    # represent the pairs of entities for relations separated by a `-` (include only one relation direction),
    # and sample_weight_or_counts should be between 0 and 1 if `negative_relation_strategy` is 'weighted' or
    # between 0 and the max number of occurrences of negative relations if `negative_relation_strategy` is
    # 'counts', defaults to None
    negative_relation_strategy_dict = {'DATE-PROBLEM': 0.1, 'TREATMENT-PROBLEM': 0.5, 'TEST-PROBLEM': 0.2},

    # optional: list of Annotation Lab task IDs to exclude from output dataframe, defaults to None
    # excluded_task_ids = [2, 3]

    # optional: list of Annotation Lab task titles to exclude from output dataframe, defaults to None
    # excluded_task_titles = ['Note 1']

)
```

{:.shell-output}

```sh
Successfully processed relations for task: Task ID# 2
Successfully processed relations for task: Task ID# 3
Successfully processed relations for task: Task ID# 1
Total tasks processed: 3
Total annotated relations processed: 10
sentence_detector_dl_healthcare download started this may take some time.
Approximate size to download 367.3 KB
[OK!]
Successfully processed NER labels for: Task ID# 2
Successfully processed NER labels for: Task ID# 3
Successfully processed NER labels for: Task ID# 1
Total tasks processed: 3
Total annotated NER labels processed: 28
```

**Output:**

<div class="table_wrapper" markdown="1">

|     | task_id | title  | sentence                                          | firstCharEnt1 | firstCharEnt2 | lastCharEnt1 | lastCharEnt2 | chunk1                      | chunk2                      | label1    | label2    | rel             |
| --- | ------- | ------ | ------------------------------------------------- | ------------- | ------------- | ------------ | ------------ | --------------------------- | --------------------------- | --------- | --------- | --------------- |
| 0   | 1       | Note 1 | On 18/08 patient declares she has a headache s... | 36            | 51            | 44           | 56           | headache                    | 06/08                       | PROBLEM   | DATE      | is_date_of      |
| 1   | 1       | Note 1 | On 18/08 patient declares she has a headache s... | 36            | 73            | 44           | 80           | headache                    | head CT                     | PROBLEM   | TEST      | is_test_of      |
| 2   | 1       | Note 1 | On 18/08 patient declares she has a headache s... | 51            | 156           | 56           | 160          | 06/08                       | pain                        | DATE      | PROBLEM   | O               |
| 3   | 1       | Note 1 | On 18/08 patient declares she has a headache s... | 73            | 126           | 80           | 134          | head CT                     | alopecia                    | TEST      | PROBLEM   | O               |
| 4   | 2       | Note 2 | At that time, physical exam showed no signs of... | 14            | 47            | 27           | 50           | physical exam               | flu                         | TEST      | PROBLEM   | is_test_of      |
| 5   | 2       | Note 2 | The patient is a 5-month-old infant who presen... | 63            | 76            | 68           | 80           | Feb 8                       | cold                        | DATE      | PROBLEM   | is_date_of      |
| 6   | 2       | Note 2 | The patient is a 5-month-old infant who presen... | 63            | 82            | 68           | 87           | Feb 8                       | cough                       | DATE      | PROBLEM   | is_date_of      |
| 7   | 2       | Note 2 | The patient is a 5-month-old infant who presen... | 63            | 93            | 68           | 103          | Feb 8                       | runny nose                  | DATE      | PROBLEM   | is_date_of      |
| 8   | 2       | Note 2 | The patient is a 5-month-old infant who presen... | 82            | 110           | 87           | 115          | cough                       | Feb 2                       | PROBLEM   | DATE      | O               |
| 9   | 3       | Note 3 | One day ago, mom also noticed a tactile temper... | 32            | 73            | 51           | 80           | tactile temperature         | Tylenol                     | PROBLEM   | TREATMENT | is_treatment_of |
| 10  | 3       | Note 3 | The patient is a 21-day-old male here for 2 da... | 52            | 69            | 62           | 77           | congestion                  | Nov 8/15                    | PROBLEM   | DATE      | is_date_of      |
| 11  | 3       | Note 3 | The patient is a 21-day-old male here for 2 da... | 52            | 93            | 62           | 120          | congestion                  | suctioning yellow discharge | PROBLEM   | TREATMENT | is_treatment_of |
| 12  | 3       | Note 3 | The patient is a 21-day-old male here for 2 da... | 93            | 244           | 120          | 261          | suctioning yellow discharge | perioral cyanosis           | TREATMENT | PROBLEM   | O               |
| 13  | 3       | Note 3 | The patient is a 21-day-old male here for 2 da... | 93            | 265           | 120          | 276          | suctioning yellow discharge | retractions                 | TREATMENT | PROBLEM   | O               |
| 14  | 3       | Note 3 | The patient is a 21-day-old male here for 2 da... | 173           | 217           | 196          | 225          | mild breathing problems     | Nov 9/15                    | PROBLEM   | DATE      | is_date_of      |

</div>

## Generate JSON containing Pre-annotations using a Spark NLP pipeline

{:.info}
No Alab Credentials required.

Define Spark NLP for Healthcare pipeline

```py
document = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentence = SentenceDetector()\
    .setInputCols(['document'])\
    .setOutputCol('sentence')\
    .setCustomBounds(['\n'])

tokenizer = Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

word_embeddings = WordEmbeddingsModel().pretrained('embeddings_clinical', 'en', 'clinical/models')\
    .setInputCols(["sentence", 'token'])\
    .setOutputCol("embeddings")\

ner_model = MedicalNerModel.pretrained('ner_jsl', 'en', 'clinical/models')\
    .setInputCols(["sentence", "token", "embeddings"])\
    .setOutputCol("ner")

converter = NerConverter()\
    .setInputCols(["sentence", "token", "ner"])\
    .setOutputCol("ner_chunk")

assertion_model = AssertionDLModel().pretrained('assertion_dl', 'en', 'clinical/models')\
    .setInputCols(["sentence", "ner_chunk", 'embeddings'])\
    .setOutputCol("assertion_res")

pos_tagger = PerceptronModel()\
    .pretrained("pos_clinical", "en", "clinical/models") \
    .setInputCols(["sentence", "token"])\
    .setOutputCol("pos_tags")

dependency_parser = DependencyParserModel()\
    .pretrained("dependency_conllu", "en")\
    .setInputCols(["sentence", "pos_tags", "token"])\
    .setOutputCol("dependencies")

relation_clinical = RelationExtractionModel.pretrained('re_clinical', 'en', 'clinical/models')\
    .setInputCols(["embeddings", "pos_tags", "ner_chunk", "dependencies"])\
    .setOutputCol("relations_clinical")\
    .setRelationPairs(['procedure-disease_syndrome_disorder', 'test-oncological', 'test-disease_syndrome_disorder',
                       'external_body_part_or_region-procedure', 'oncological-external_body_part_or_region',
                       'oncological-procedure'])\
    .setMaxSyntacticDistance(0)

relation_pos = RelationExtractionModel.pretrained('posology_re', 'en', 'clinical/models')\
    .setInputCols(["embeddings", "pos_tags", "ner_chunk", "dependencies"])\
    .setOutputCol("relations_pos")\
    .setRelationPairs(['drug_ingredient-drug_brandname', 'drug_ingredient-dosage', 'drug_ingredient-strength', 'drug_ingredient-route'])\
    .setMaxSyntacticDistance(0)

ner_pipeline = Pipeline(
    stages = [
        document,
        sentence,
        tokenizer,
        word_embeddings,
        ner_model,
        converter,
        assertion_model,
        pos_tagger,
        dependency_parser,
        relation_clinical,
        relation_pos
    ])

empty_data = spark.createDataFrame([['']]).toDF("text")
pipeline_model = ner_pipeline.fit(empty_data)
lmodel = LightPipeline(pipeline_model)
```

{:.shell-output}

```sh
embeddings_clinical download started this may take some time.
Approximate size to download 1.6 GB
[OK!]
ner_jsl download started this may take some time.
[OK!]
assertion_dl download started this may take some time.
[OK!]
pos_clinical download started this may take some time.
Approximate size to download 1.5 MB
[OK!]
dependency_conllu download started this may take some time.
Approximate size to download 16.7 MB
[OK!]
re_clinical download started this may take some time.
Approximate size to download 6 MB
[OK!]
```

**Run on sample tasks**

```py
txt1 = "The patient is a 21-day-old male here for 2 days of congestion since Nov 8/15 - mom has been suctioning yellow discharge from the patient's nares, plus she has noticed some mild breathing problems while feeding since Nov 9/15 (without signs of perioral cyanosis or retractions). One day ago, mom also noticed a tactile temperature and gave the patient Tylenol."
txt2 = "The patient is a 5-month-old infant who presented initially on Feb 8 with a cold, cough, and runny nose since Feb 2. Mom states she had no fever. She had no difficulty breathing and her cough was described as dry and hacky. At that time, physical exam showed no signs of flu."
task_list = [txt1, txt2]

results = lmodel.fullAnnotate(task_list)

# full pipeline:
# results = pipeline_model.transform(spark.createDataFrame(pd.DataFrame({'text': task_list}))).collect()
```

**Generate Pre-annotation JSON using pipeline results**

```py
pre_annotations, summary = alab.generate_preannotations(

    # required: list of results.
    all_results = results,

    # requied: output column name of 'DocumentAssembler' stage - to get original document string.
    document_column = 'document',

    # required: column name(s) of ner model(s). Note: multiple NER models can be used, but make sure their results don't overrlap.
    # Or use 'ChunkMergeApproach' to combine results from multiple NER models.
    ner_columns = ['ner_chunk'],

    # optional: column name(s) of assertion model(s). Note: multiple assertion models can be used, but make sure their results don't overrlap.
    assertion_columns = ['assertion_res'],

    # optional: column name(s) of relation extraction model(s). Note: multiple relation extraction models can be used, but make sure their results don't overrlap.
    relations_columns = ['relations_clinical', 'relations_pos'],

    # optional: This can be defined to identify which pipeline/user/model was used to get predictions.
    # Default: 'model'
    user_name = 'model',

    # optional: Option to assign custom titles to tasks. By default, tasks will be titled as 'task_#'
    titles_list = [],

    # optional: If there are already tasks in project, then this id offset can be used to make sure default titles 'task_#' do not overlap.
    # While upload a batch after the first one, this can be set to number of tasks currently present in the project
    # This number would be added to each tasks's ID and title.
    id_offset=0

)
```

{:.shell-output}

```sh
Processing 2 Annotations.
```

Generated JSON. This can be uploaded directly via UI or via [API](https://nlp.johnsnowlabs.com/docs/en/alab/healthcare#upload-pre-annotations-to-annotation-lab).

```sh
pre_annotations
```

{:.shell-output}

```sh
[{'predictions': [{'created_username': 'model',
    'result': [{'from_name': 'label',
      'id': 'YCtU7EDvme',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 27,
       'labels': ['Age'],
       'start': 17,
       'text': '21-day-old'}},
     {'from_name': 'label',
      'id': 'xqbYIUPhhB',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 32, 'labels': ['Gender'], 'start': 28, 'text': 'male'}},
     {'from_name': 'label',
      'id': '7GYr3DFbAs',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 48,
       'labels': ['Duration'],
       'start': 38,
       'text': 'for 2 days'}},
     {'from_name': 'label',
      'id': 'akBx3N0Gy2',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 62,
       'labels': ['Symptom'],
       'start': 52,
       'text': 'congestion'}},
     {'from_name': 'label',
      'id': 'TJKowx9hR2',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 77,
       'labels': ['Date'],
       'start': 69,
       'text': 'Nov 8/15'}},
     {'from_name': 'label',
      'id': 'UuWHo6pGz8',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 83, 'labels': ['Gender'], 'start': 80, 'text': 'mom'}},
     {'from_name': 'label',
      'id': 'qIgnDgSJw6',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 110,
       'labels': ['Modifier'],
       'start': 104,
       'text': 'yellow'}},
     {'from_name': 'label',
      'id': 'DkE8rIoKVg',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 120,
       'labels': ['Symptom'],
       'start': 111,
       'text': 'discharge'}},
     {'from_name': 'label',
      'id': 'RBjrHSa1sj',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 145,
       'labels': ['External_body_part_or_region'],
       'start': 140,
       'text': 'nares'}},
     {'from_name': 'label',
      'id': 'yHEPWvrk9s',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 155,
       'labels': ['Gender'],
       'start': 152,
       'text': 'she'}},
     {'from_name': 'label',
      'id': 'dbeel0WXqw',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 177,
       'labels': ['Modifier'],
       'start': 173,
       'text': 'mild'}},
     {'from_name': 'label',
      'id': 'cFJwsYMe2k',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 210,
       'labels': ['Symptom'],
       'start': 178,
       'text': 'breathing problems while feeding'}},
     {'from_name': 'label',
      'id': 'PhiSDTDXlV',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 225,
       'labels': ['Date'],
       'start': 217,
       'text': 'Nov 9/15'}},
     {'from_name': 'label',
      'id': 'lsGep4SLRn',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 261,
       'labels': ['Symptom'],
       'start': 244,
       'text': 'perioral cyanosis'}},
     {'from_name': 'label',
      'id': 'WiTJIGOZ9Z',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 276,
       'labels': ['Symptom'],
       'start': 265,
       'text': 'retractions'}},
     {'from_name': 'label',
      'id': 'omIdHl5z74',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 290,
       'labels': ['RelativeDate'],
       'start': 279,
       'text': 'One day ago'}},
     {'from_name': 'label',
      'id': 'CqDclquhmD',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 295,
       'labels': ['Gender'],
       'start': 292,
       'text': 'mom'}},
     {'from_name': 'label',
      'id': 'u8Q3GTVzZh',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 359,
       'labels': ['Drug_BrandName'],
       'start': 352,
       'text': 'Tylenol'}},
     {'from_name': 'label',
      'id': 'i2JLPQOUxv',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 27, 'labels': ['absent'], 'start': 17, 'text': 'Age'}},
     {'from_name': 'label',
      'id': 'QShr4s6bpg',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 32,
       'labels': ['present'],
       'start': 28,
       'text': 'Gender'}},
     {'from_name': 'label',
      'id': '800DYq0quS',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 48,
       'labels': ['present'],
       'start': 38,
       'text': 'Duration'}},
     {'from_name': 'label',
      'id': 'ns3P70kktN',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 62,
       'labels': ['present'],
       'start': 52,
       'text': 'Symptom'}},
     {'from_name': 'label',
      'id': 'tQOdI1ANUO',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 77,
       'labels': ['present'],
       'start': 69,
       'text': 'Date'}},
     {'from_name': 'label',
      'id': 'oEwYVnyi2A',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 83,
       'labels': ['present'],
       'start': 80,
       'text': 'Gender'}},
     {'from_name': 'label',
      'id': 'PkCkXQEIFN',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 110,
       'labels': ['present'],
       'start': 104,
       'text': 'Modifier'}},
     {'from_name': 'label',
      'id': 'm9Bz8CzaXd',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 120,
       'labels': ['present'],
       'start': 111,
       'text': 'Symptom'}},
     {'from_name': 'label',
      'id': 'GBIhXo8nks',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 145,
       'labels': ['absent'],
       'start': 140,
       'text': 'External_body_part_or_region'}},
     {'from_name': 'label',
      'id': 'CDVDDIwVrl',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 155,
       'labels': ['absent'],
       'start': 152,
       'text': 'Gender'}},
     {'from_name': 'label',
      'id': 'CHPuFoT9iK',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 177,
       'labels': ['present'],
       'start': 173,
       'text': 'Modifier'}},
     {'from_name': 'label',
      'id': 'eS4vXGth7v',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 210,
       'labels': ['present'],
       'start': 178,
       'text': 'Symptom'}},
     {'from_name': 'label',
      'id': '2nDgfsoGZ7',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 225,
       'labels': ['present'],
       'start': 217,
       'text': 'Date'}},
     {'from_name': 'label',
      'id': 'dhYC0U4sKG',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 261,
       'labels': ['absent'],
       'start': 244,
       'text': 'Symptom'}},
     {'from_name': 'label',
      'id': 'TKN6DIP2ua',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 276,
       'labels': ['absent'],
       'start': 265,
       'text': 'Symptom'}},
     {'from_name': 'label',
      'id': '7X9EwULTA1',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 290,
       'labels': ['present'],
       'start': 279,
       'text': 'RelativeDate'}},
     {'from_name': 'label',
      'id': 'UIFKYTDKcm',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 295,
       'labels': ['present'],
       'start': 292,
       'text': 'Gender'}},
     {'from_name': 'label',
      'id': 'U6TqOYf3Ez',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 359,
       'labels': ['present'],
       'start': 352,
       'text': 'Drug_BrandName'}}]}],
  'data': {'title': 'task_0',
   'text': "The patient is a 21-day-old male here for 2 days of congestion since Nov 8/15 - mom has been suctioning yellow discharge from the patient's nares, plus she has noticed some mild breathing problems while feeding since Nov 9/15 (without signs of perioral cyanosis or retractions). One day ago, mom also noticed a tactile temperature and gave the patient Tylenol."},
  'id': 0},
 {'predictions': [{'created_username': 'model',
    'result': [{'from_name': 'label',
      'id': 'qd28OkdmDO',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 28,
       'labels': ['Age'],
       'start': 17,
       'text': '5-month-old'}},
     {'from_name': 'label',
      'id': 'UIZm8wCy3c',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 35, 'labels': ['Age'], 'start': 29, 'text': 'infant'}},
     {'from_name': 'label',
      'id': 'KpMv4PIy21',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 68, 'labels': ['Date'], 'start': 63, 'text': 'Feb 8'}},
     {'from_name': 'label',
      'id': 'Uyj3awC8jp',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 80,
       'labels': ['Symptom'],
       'start': 76,
       'text': 'cold'}},
     {'from_name': 'label',
      'id': 'Dt3xtm1l5A',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 87,
       'labels': ['Symptom'],
       'start': 82,
       'text': 'cough'}},
     {'from_name': 'label',
      'id': 'bp9yUFAUaE',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 103,
       'labels': ['Symptom'],
       'start': 93,
       'text': 'runny nose'}},
     {'from_name': 'label',
      'id': 'QhuFKxwFVk',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 113, 'labels': ['Date'], 'start': 110, 'text': 'Feb'}},
     {'from_name': 'label',
      'id': 'm9ikgaeJMY',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 120,
       'labels': ['Gender'],
       'start': 117,
       'text': 'Mom'}},
     {'from_name': 'label',
      'id': 'QXhhDJ6CXn',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 131,
       'labels': ['Gender'],
       'start': 128,
       'text': 'she'}},
     {'from_name': 'label',
      'id': 'YUCHE7GcHB',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 144,
       'labels': ['VS_Finding'],
       'start': 139,
       'text': 'fever'}},
     {'from_name': 'label',
      'id': 'xbphfajGY1',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 149,
       'labels': ['Gender'],
       'start': 146,
       'text': 'She'}},
     {'from_name': 'label',
      'id': 'xN5GuZpeUw',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 177,
       'labels': ['Symptom'],
       'start': 157,
       'text': 'difficulty breathing'}},
     {'from_name': 'label',
      'id': 'VK9lAjcVNy',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 185,
       'labels': ['Gender'],
       'start': 182,
       'text': 'her'}},
     {'from_name': 'label',
      'id': 'dqiohcfX4G',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 191,
       'labels': ['Symptom'],
       'start': 186,
       'text': 'cough'}},
     {'from_name': 'label',
      'id': '18bjvjxuDL',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 212,
       'labels': ['Modifier'],
       'start': 209,
       'text': 'dry'}},
     {'from_name': 'label',
      'id': 'BB90sIXIYZ',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 274,
       'labels': ['Disease_Syndrome_Disorder'],
       'start': 271,
       'text': 'flu'}},
     {'from_name': 'label',
      'id': 'sAL9AHWvOa',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 28, 'labels': ['absent'], 'start': 17, 'text': 'Age'}},
     {'from_name': 'label',
      'id': 'vyio7vnpmS',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 35, 'labels': ['present'], 'start': 29, 'text': 'Age'}},
     {'from_name': 'label',
      'id': 'r6T6e8WmO9',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 68,
       'labels': ['present'],
       'start': 63,
       'text': 'Date'}},
     {'from_name': 'label',
      'id': '3SdFeft6ya',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 80,
       'labels': ['present'],
       'start': 76,
       'text': 'Symptom'}},
     {'from_name': 'label',
      'id': '0iyfhRx1nl',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 87,
       'labels': ['present'],
       'start': 82,
       'text': 'Symptom'}},
     {'from_name': 'label',
      'id': 'pqJFZRu8Zu',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 103,
       'labels': ['present'],
       'start': 93,
       'text': 'Symptom'}},
     {'from_name': 'label',
      'id': 'zRa9noedl5',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 113,
       'labels': ['present'],
       'start': 110,
       'text': 'Date'}},
     {'from_name': 'label',
      'id': 'RJ8MHb5Css',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 120,
       'labels': ['absent'],
       'start': 117,
       'text': 'Gender'}},
     {'from_name': 'label',
      'id': 'sbtQpMnkxH',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 131,
       'labels': ['absent'],
       'start': 128,
       'text': 'Gender'}},
     {'from_name': 'label',
      'id': 'K0yEKeG7GR',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 144,
       'labels': ['absent'],
       'start': 139,
       'text': 'VS_Finding'}},
     {'from_name': 'label',
      'id': 'V4fTVAh4Ro',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 149,
       'labels': ['absent'],
       'start': 146,
       'text': 'Gender'}},
     {'from_name': 'label',
      'id': '1K1NUt9mcU',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 177,
       'labels': ['absent'],
       'start': 157,
       'text': 'Symptom'}},
     {'from_name': 'label',
      'id': 'kXl3bnMSqM',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 185,
       'labels': ['absent'],
       'start': 182,
       'text': 'Gender'}},
     {'from_name': 'label',
      'id': 'spqjsrISZg',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 191,
       'labels': ['present'],
       'start': 186,
       'text': 'Symptom'}},
     {'from_name': 'label',
      'id': 'EcrKDs2yyH',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 212,
       'labels': ['present'],
       'start': 209,
       'text': 'Modifier'}},
     {'from_name': 'label',
      'id': 'FHYcyz14aj',
      'source': '$text',
      'to_name': 'text',
      'type': 'labels',
      'value': {'end': 274,
       'labels': ['absent'],
       'start': 271,
       'text': 'Disease_Syndrome_Disorder'}}]}],
  'data': {'title': 'task_1',
   'text': 'The patient is a 5-month-old infant who presented initially on Feb 8 with a cold, cough, and runny nose since Feb 2. Mom states she had no fever. She had no difficulty breathing and her cough was described as dry and hacky. At that time, physical exam showed no signs of flu.'},
  'id': 1}]
```

A Summary is also generated. This summary can be used to [setup project configuration](https://nlp.johnsnowlabs.com/docs/en/alab/healthcare#set-configuration-using-summary-generated-at-the-pre-annotation-step).

{:.shell-output}

```sh
{
  'ner_labels': [
    'Age',
    'VS_Finding',
    'Gender',
    'Modifier',
    'Duration',
    'RelativeDate',
    'Symptom',
    'Date',
    'External_body_part_or_region',
    'Disease_Syndrome_Disorder',
    'Drug_BrandName'
  ],
  'assertion_labels': ['present', 'absent'],
  're_labels': []
}
```

## Interacting with Annotation Lab

{:.info}
Credentials required

**Set Credentials**

```py
alab = AnnotationLab()

username=''
password=''
client_secret=""
annotationlab_url=""

alab.set_credentials(

    # required: username
    username=username,

    # required: password
    password=password,

    # required: secret for you alab instance (every alab installation has a different secret)
    client_secret=client_secret,

    # required: http(s) url for you annotation lab
    annotationlab_url=annotationlab_url

    )
```

<br />

### Get All visible projects

```py
alab.get_all_projects()
```

{:.shell-output}

```sh
Operation completed successfully. Response code: 200
{'has_next': True,
 'has_prev': False,
 'items': [{'creation_date': 'Thu, 29 Sep 2022 18:01:07 GMT',
   'group_color': None,
   'group_id': None,
   'group_name': None,
   'owner': 'hasham',
   'owner_id': 'ba60df4b-7192-47ca-aa92-759fa577a617',
   'project_description': '',
   'project_id': 1129,
   'project_members': ['hasham'],
   'project_name': 'alab_demo',
   'resource_id': '1dabaac8-54a0-4c52-a876-8c01f42b44e7',
   'total_tasks': 2},
  {'creation_date': 'Tue, 27 Sep 2022 03:12:18 GMT',
   'group_color': None,
   'group_id': None,
   'group_name': None,
   'owner': 'hasham',
   'owner_id': 'ba60df4b-7192-47ca-aa92-759fa577a617',
   'project_description': '',
   'project_id': 1117,
   'project_members': ['hasham'],
   'project_name': 'testing101',
   'resource_id': 'b1388775-9a3b-436e-b1cc-ea36bab44699',
   'total_tasks': 9},
  {'creation_date': 'Tue, 27 Sep 2022 03:08:21 GMT',
   'group_color': None,
   'group_id': None,
   'group_name': None,
   'owner': 'hasham',
   'owner_id': 'ba60df4b-7192-47ca-aa92-759fa577a617',
   'project_description': '',
   'project_id': 1116,
   'project_members': ['hasham'],
   'project_name': 'testing1',
   'resource_id': '3340d328-91b5-47a5-b694-502a949b1367',
   'total_tasks': 0},
  {'creation_date': 'Sun, 04 Sep 2022 00:15:56 GMT',
   'group_color': None,
   'group_id': None,
   'group_name': None,
   'owner': 'hasham',
   'owner_id': 'ba60df4b-7192-47ca-aa92-759fa577a617',
   'project_description': '',
   'project_id': 1053,
   'project_members': ['hasham', 'halil.saglamlar'],
   'project_name': 'dummy',
   'resource_id': '2aff5afc-7ef3-47d5-b84d-aed3e062a8d5',
   'total_tasks': 20},
  {'creation_date': 'Wed, 27 Jul 2022 16:14:11 GMT',
   'group_color': None,
   'group_id': None,
   'group_name': None,
   'owner': 'luca.martial',
   'owner_id': 'e104d8b1-15d7-4a30-97df-1344a8c3c52f',
   'project_description': '',
   'project_id': 970,
   'project_members': ['luca.martial', 'hasham'],
   'project_name': 'alab_module',
   'resource_id': '0e07ae56-29f8-4c56-906a-4abb93848381',
   'total_tasks': 3},
  {'creation_date': 'Fri, 15 Jul 2022 14:05:55 GMT',
   'group_color': None,
   'group_id': None,
   'group_name': None,
   'owner': 'hasham',
   'owner_id': 'ba60df4b-7192-47ca-aa92-759fa577a617',
   'project_description': '',
   'project_id': 957,
   'project_members': ['hasham', 'jay', 'nabin'],
   'project_name': 'RE_ADE_Conversational',
   'resource_id': 'c762134f-8467-4248-ba48-c88f864e742f',
   'total_tasks': 1045},
  {'creation_date': 'Mon, 25 Apr 2022 08:16:11 GMT',
   'group_color': None,
   'group_id': None,
   'group_name': None,
   'owner': 'ciprian',
   'owner_id': '54b92e23-a857-4e76-b0e4-b49e085c4614',
   'project_description': '',
   'project_id': 841,
   'project_members': ['pranab', 'nabin', 'vijaya@kognitic.com'],
   'project_name': 'Kognitic',
   'resource_id': 'a4fc6141-39aa-4ed0-82be-fcffc641c4df',
   'total_tasks': 102},
  {'creation_date': 'Mon, 11 Apr 2022 17:55:09 GMT',
   'group_color': None,
   'group_id': None,
   'group_name': None,
   'owner': 'hasham',
   'owner_id': 'ba60df4b-7192-47ca-aa92-759fa577a617',
   'project_description': '',
   'project_id': 819,
   'project_members': [],
   'project_name': 'bms_citeline_ph3',
   'resource_id': '6fcb9e65-045d-43b6-94dd-8256da7ed3d6',
   'total_tasks': 674},
  {'creation_date': 'Tue, 22 Mar 2022 19:00:20 GMT',
   'group_color': None,
   'group_id': None,
   'group_name': None,
   'owner': 'hasham',
   'owner_id': 'ba60df4b-7192-47ca-aa92-759fa577a617',
   'project_description': '',
   'project_id': 790,
   'project_members': [],
   'project_name': 'lot_demo_bms',
   'resource_id': '8b136ab6-aab5-468e-830a-251fef3be528',
   'total_tasks': 11},
  {'creation_date': 'Tue, 22 Mar 2022 03:47:53 GMT',
   'group_color': None,
   'group_id': None,
   'group_name': None,
   'owner': 'hasham',
   'owner_id': 'ba60df4b-7192-47ca-aa92-759fa577a617',
   'project_description': '',
   'project_id': 788,
   'project_members': [],
   'project_name': 'bms_citeline',
   'resource_id': '0a229a5f-f055-412f-bc5c-2c70e43b93a1',
   'total_tasks': 674},
  {'creation_date': 'Mon, 17 Jan 2022 19:02:42 GMT',
   'group_color': None,
   'group_id': None,
   'group_name': None,
   'owner': 'hasham',
   'owner_id': 'ba60df4b-7192-47ca-aa92-759fa577a617',
   'project_description': '',
   'project_id': 681,
   'project_members': [],
   'project_name': 'Internal_NER_JSL_Annotations',
   'resource_id': '8c7f4ffb-ce3d-44cc-b482-a0edce3688d7',
   'total_tasks': 1000},
  {'creation_date': 'Mon, 17 Jan 2022 14:56:57 GMT',
   'group_color': None,
   'group_id': None,
   'group_name': None,
   'owner': 'pranab',
   'owner_id': 'daa1630c-0cbb-4e32-855c-6e351a0ebef6',
   'project_description': '',
   'project_id': 679,
   'project_members': ['amit', 'angel'],
   'project_name': 'NER_JSL_PRANAB_CLONE',
   'resource_id': '530642ab-6715-4647-a910-1287662933fd',
   'total_tasks': 299},
  {'creation_date': 'Wed, 12 Jan 2022 15:01:58 GMT',
   'group_color': None,
   'group_id': None,
   'group_name': None,
   'owner': 'hasham',
   'owner_id': 'ba60df4b-7192-47ca-aa92-759fa577a617',
   'project_description': '',
   'project_id': 673,
   'project_members': [],
   'project_name': 'bms_demo',
   'resource_id': '55b1052b-5cf5-4a9d-b6b7-00a5e516e18d',
   'total_tasks': 206},
  {'creation_date': 'Wed, 29 Dec 2021 17:40:35 GMT',
   'group_color': None,
   'group_id': None,
   'group_name': None,
   'owner': 'hasham',
   'owner_id': 'ba60df4b-7192-47ca-aa92-759fa577a617',
   'project_description': '',
   'project_id': 657,
   'project_members': [],
   'project_name': 'Internal_Abbreviation_Project',
   'resource_id': 'a0ad3b4d-bff2-4238-b9f1-db16f0d72967',
   'total_tasks': 2847},
  {'creation_date': 'Wed, 01 Dec 2021 12:31:53 GMT',
   'group_color': None,
   'group_id': None,
   'group_name': None,
   'owner': 'mauro',
   'owner_id': '7b6048c8-f923-46e4-9011-2c749e3c2c93',
   'project_description': '',
   'project_id': 634,
   'project_members': [],
   'project_name': 'COVID_Clinical_Trials_Round_2',
   'resource_id': '80ac9546-9344-4b71-ad90-7252d20b7d94',
   'total_tasks': 0},
  {'creation_date': 'Fri, 26 Nov 2021 21:42:02 GMT',
   'group_color': None,
   'group_id': None,
   'group_name': None,
   'owner': 'hasham',
   'owner_id': 'ba60df4b-7192-47ca-aa92-759fa577a617',
   'project_description': '',
   'project_id': 632,
   'project_members': ['luca.martial'],
   'project_name': 'demo_100',
   'resource_id': '54c33f63-f56e-4d30-ac8e-ccce036f31ae',
   'total_tasks': 50},
  {'creation_date': 'Tue, 02 Nov 2021 21:55:59 GMT',
   'group_color': None,
   'group_id': None,
   'group_name': None,
   'owner': 'hasham',
   'owner_id': 'ba60df4b-7192-47ca-aa92-759fa577a617',
   'project_description': '',
   'project_id': 616,
   'project_members': [],
   'project_name': 'Optum_PhaseI_MTSamples',
   'resource_id': '45bbc554-94c7-4750-a716-a8e36923475a',
   'total_tasks': 0},
  {'creation_date': 'Mon, 25 Oct 2021 19:23:50 GMT',
   'group_color': None,
   'group_id': None,
   'group_name': None,
   'owner': 'ahmet',
   'owner_id': '96c249a7-8496-482b-834e-3cc2417c3df6',
   'project_description': '',
   'project_id': 605,
   'project_members': [],
   'project_name': 'annotate',
   'resource_id': 'a6b52d97-7735-4f65-98b0-f9801801f8af',
   'total_tasks': 3},
  {'creation_date': 'Tue, 05 Oct 2021 22:22:20 GMT',
   'group_color': None,
   'group_id': None,
   'group_name': None,
   'owner': 'hasham',
   'owner_id': 'ba60df4b-7192-47ca-aa92-759fa577a617',
   'project_description': '',
   'project_id': 575,
   'project_members': [],
   'project_name': 'BMS_Drug_Dev_Endpoint',
   'resource_id': 'dbd636dc-3e0f-43a7-95e1-59bbe7d5aa13',
   'total_tasks': 200},
  {'creation_date': 'Mon, 13 Sep 2021 08:14:37 GMT',
   'group_color': None,
   'group_id': None,
   'group_name': None,
   'owner': 'hasham',
   'owner_id': 'ba60df4b-7192-47ca-aa92-759fa577a617',
   'project_description': '',
   'project_id': 543,
   'project_members': [],
   'project_name': 'BMS_drug_dev',
   'resource_id': '460be549-9df6-40b7-88ac-76957bd1b532',
   'total_tasks': 1101},
  {'creation_date': 'Tue, 24 Aug 2021 18:21:20 GMT',
   'group_color': None,
   'group_id': None,
   'group_name': None,
   'owner': 'hasham',
   'owner_id': 'ba60df4b-7192-47ca-aa92-759fa577a617',
   'project_description': '',
   'project_id': 513,
   'project_members': [],
   'project_name': 'BMS_Annotation_Diag_track_2',
   'resource_id': '7d7fbc76-bb4b-4e46-b0d5-783817b9c1b7',
   'total_tasks': 958},
  {'creation_date': 'Tue, 17 Aug 2021 15:44:44 GMT',
   'group_color': None,
   'group_id': None,
   'group_name': None,
   'owner': 'hasham',
   'owner_id': 'ba60df4b-7192-47ca-aa92-759fa577a617',
   'project_description': '',
   'project_id': 509,
   'project_members': [],
   'project_name': 'BMS_Annotation_Trt_Track_cycle_treatment_fix',
   'resource_id': 'a3fc5c30-368b-4620-9038-ceefa60aa3c8',
   'total_tasks': 303},
  {'creation_date': 'Thu, 12 Aug 2021 13:56:16 GMT',
   'group_color': None,
   'group_id': None,
   'group_name': None,
   'owner': 'hasham',
   'owner_id': 'ba60df4b-7192-47ca-aa92-759fa577a617',
   'project_description': '',
   'project_id': 507,
   'project_members': [],
   'project_name': 'BMS_Annotation_Diag_track',
   'resource_id': '8393a9c7-8fd4-4abe-9933-324ea6a1c13d',
   'total_tasks': 958},
  {'creation_date': 'Thu, 12 Aug 2021 13:46:11 GMT',
   'group_color': None,
   'group_id': None,
   'group_name': None,
   'owner': 'hasham',
   'owner_id': 'ba60df4b-7192-47ca-aa92-759fa577a617',
   'project_description': '',
   'project_id': 506,
   'project_members': [],
   'project_name': 'BMS_Test',
   'resource_id': 'd07562c8-799d-4b4b-832f-0f21c836f7e2',
   'total_tasks': 1074},
  {'creation_date': 'Sat, 17 Jul 2021 08:14:50 GMT',
   'group_color': None,
   'group_id': None,
   'group_name': None,
   'owner': 'hasham',
   'owner_id': 'ba60df4b-7192-47ca-aa92-759fa577a617',
   'project_description': '',
   'project_id': 491,
   'project_members': ['luca.martial'],
   'project_name': 'Internal_COVID_Clinical_Trials_Data',
   'resource_id': 'c3254044-98f9-43eb-8eee-ece3f44963fd',
   'total_tasks': 100},
  {'creation_date': 'Mon, 05 Jul 2021 16:22:25 GMT',
   'group_color': None,
   'group_id': None,
   'group_name': None,
   'owner': 'hasham',
   'owner_id': 'ba60df4b-7192-47ca-aa92-759fa577a617',
   'project_description': '',
   'project_id': 486,
   'project_members': [],
   'project_name': 'Internal_Biomarkers_Pubmed',
   'resource_id': 'add20481-358b-46dd-b798-e5d71067d262',
   'total_tasks': 200},
  {'creation_date': 'Sat, 03 Jul 2021 15:25:40 GMT',
   'group_color': None,
   'group_id': None,
   'group_name': None,
   'owner': 'hasham',
   'owner_id': 'ba60df4b-7192-47ca-aa92-759fa577a617',
   'project_description': "This is a combined project of GE's annotations for phase 1.  It contains Train, Dev, Test set, as well as the new data they provided.",
   'project_id': 485,
   'project_members': [],
   'project_name': 'GE_PhaseI_Combined_All_Sets',
   'resource_id': '5afd962a-a37c-4be3-86da-c0115cfa2807',
   'total_tasks': 52},
  {'creation_date': 'Sat, 03 Jul 2021 14:55:41 GMT',
   'group_color': None,
   'group_id': None,
   'group_name': None,
   'owner': 'hasham',
   'owner_id': 'ba60df4b-7192-47ca-aa92-759fa577a617',
   'project_description': "These are sentences sampled from JSL WIP Ner. The main purpose was to merge it with customer's data to make the model more robust. Since the AG is different and this data is already annotated for JSL WIP , this data is not to be used with that model. It can have other uses.",
   'project_id': 484,
   'project_members': [],
   'project_name': 'GE_Project_JSL_WIP_Fixed_by_Andrie_for_GE_BP_Only',
   'resource_id': '948878a2-480e-4ce0-8e75-6fde90a32d6e',
   'total_tasks': 50},
  {'creation_date': 'Sun, 03 Jan 2021 22:53:47 GMT',
   'group_color': None,
   'group_id': None,
   'group_name': None,
   'owner': 'veysel',
   'owner_id': '7803c2e3-c0cc-4463-a0cb-bbb103b2a55e',
   'project_description': '',
   'project_id': 360,
   'project_members': [],
   'project_name': 'Radiology_Project_15_mt',
   'resource_id': 'bd250266-cab2-4684-a13e-4b322f2e9929',
   'total_tasks': 73},
  {'creation_date': 'Thu, 31 Dec 2020 14:27:40 GMT',
   'group_color': None,
   'group_id': None,
   'group_name': None,
   'owner': 'veysel',
   'owner_id': '7803c2e3-c0cc-4463-a0cb-bbb103b2a55e',
   'project_description': '',
   'project_id': 353,
   'project_members': [],
   'project_name': 'Radiology_Project_14_mt',
   'resource_id': '8493c018-e085-4515-952d-79cb6c7875dc',
   'total_tasks': 100},
  {'creation_date': 'Thu, 31 Dec 2020 14:19:52 GMT',
   'group_color': None,
   'group_id': None,
   'group_name': None,
   'owner': 'veysel',
   'owner_id': '7803c2e3-c0cc-4463-a0cb-bbb103b2a55e',
   'project_description': '',
   'project_id': 352,
   'project_members': [],
   'project_name': 'Radiology_Project_13_mt',
   'resource_id': 'f9d1bbd5-63f5-49a2-b459-3a1739df36dc',
   'total_tasks': 98},
  {'creation_date': 'Mon, 21 Dec 2020 21:05:57 GMT',
   'group_color': None,
   'group_id': None,
   'group_name': None,
   'owner': 'veysel',
   'owner_id': '7803c2e3-c0cc-4463-a0cb-bbb103b2a55e',
   'project_description': '',
   'project_id': 317,
   'project_members': [],
   'project_name': 'Radiology_Project_7_cxr',
   'resource_id': 'f839ca19-fb43-417b-a046-d2cf5b5f916b',
   'total_tasks': 120},
  {'creation_date': 'Mon, 21 Dec 2020 19:00:00 GMT',
   'group_color': None,
   'group_id': None,
   'group_name': None,
   'owner': 'veysel',
   'owner_id': '7803c2e3-c0cc-4463-a0cb-bbb103b2a55e',
   'project_description': '',
   'project_id': 316,
   'project_members': [],
   'project_name': 'Radiology_Project_12_cxr',
   'resource_id': '07a98fbd-51aa-43dd-81b5-33f8b9686173',
   'total_tasks': 180},
  {'creation_date': 'Mon, 21 Dec 2020 18:58:05 GMT',
   'group_color': None,
   'group_id': None,
   'group_name': None,
   'owner': 'veysel',
   'owner_id': '7803c2e3-c0cc-4463-a0cb-bbb103b2a55e',
   'project_description': '',
   'project_id': 315,
   'project_members': [],
   'project_name': 'Radiology_Project_11_cxr',
   'resource_id': '1fe15547-0716-4ab7-9fc0-cec5eea82838',
   'total_tasks': 120},
  {'creation_date': 'Mon, 21 Dec 2020 18:56:50 GMT',
   'group_color': None,
   'group_id': None,
   'group_name': None,
   'owner': 'veysel',
   'owner_id': '7803c2e3-c0cc-4463-a0cb-bbb103b2a55e',
   'project_description': '',
   'project_id': 314,
   'project_members': [],
   'project_name': 'Radiology_Project_10_cxr',
   'resource_id': 'e6131f4b-47d0-4f8b-a9f1-6d6284b0a30d',
   'total_tasks': 120},
  {'creation_date': 'Mon, 21 Dec 2020 18:54:09 GMT',
   'group_color': None,
   'group_id': None,
   'group_name': None,
   'owner': 'veysel',
   'owner_id': '7803c2e3-c0cc-4463-a0cb-bbb103b2a55e',
   'project_description': '',
   'project_id': 313,
   'project_members': [],
   'project_name': 'Radiology_Project_9_cxr',
   'resource_id': 'af8099cc-84d9-45fb-a444-06c2ae9e7541',
   'total_tasks': 120},
  {'creation_date': 'Mon, 21 Dec 2020 16:56:26 GMT',
   'group_color': None,
   'group_id': None,
   'group_name': None,
   'owner': 'veysel',
   'owner_id': '7803c2e3-c0cc-4463-a0cb-bbb103b2a55e',
   'project_description': '',
   'project_id': 312,
   'project_members': [],
   'project_name': 'Radiology_Project_8_cxr',
   'resource_id': 'b98ca5cf-310b-4b5b-a15b-87944b8be749',
   'total_tasks': 120},
  {'creation_date': 'Mon, 21 Dec 2020 16:46:24 GMT',
   'group_color': None,
   'group_id': None,
   'group_name': None,
   'owner': 'veysel',
   'owner_id': '7803c2e3-c0cc-4463-a0cb-bbb103b2a55e',
   'project_description': '',
   'project_id': 310,
   'project_members': [],
   'project_name': 'Radiology_Project_6_cxr',
   'resource_id': '454c0b9b-b595-4a43-b178-3bff36767b4f',
   'total_tasks': 120},
  {'creation_date': 'Mon, 21 Dec 2020 15:43:58 GMT',
   'group_color': '#fd8c0a',
   'group_id': 9,
   'group_name': 'Radiology_Project',
   'owner': 'mauro',
   'owner_id': '7b6048c8-f923-46e4-9011-2c749e3c2c93',
   'project_description': '',
   'project_id': 309,
   'project_members': [],
   'project_name': 'Radiology_Project_5_cxr',
   'resource_id': '06f6b178-dc4a-472c-aee6-c93ad10b0e84',
   'total_tasks': 120},
  {'creation_date': 'Mon, 07 Dec 2020 00:49:28 GMT',
   'group_color': None,
   'group_id': None,
   'group_name': None,
   'owner': 'veysel',
   'owner_id': '7803c2e3-c0cc-4463-a0cb-bbb103b2a55e',
   'project_description': '',
   'project_id': 291,
   'project_members': [],
   'project_name': 'Radiology_Project_4_meas',
   'resource_id': '2325594c-29f1-4d4c-ab07-fa7307e0dd47',
   'total_tasks': 84},
  {'creation_date': 'Wed, 02 Dec 2020 19:55:39 GMT',
   'group_color': None,
   'group_id': None,
   'group_name': None,
   'owner': 'veysel',
   'owner_id': '7803c2e3-c0cc-4463-a0cb-bbb103b2a55e',
   'project_description': '',
   'project_id': 288,
   'project_members': [],
   'project_name': 'Radiology_Project_2_ge_text',
   'resource_id': 'ad4a0e9e-564f-46db-9f73-82ae538802aa',
   'total_tasks': 40},
  {'creation_date': 'Wed, 02 Dec 2020 19:04:10 GMT',
   'group_color': None,
   'group_id': None,
   'group_name': None,
   'owner': 'veysel',
   'owner_id': '7803c2e3-c0cc-4463-a0cb-bbb103b2a55e',
   'project_description': '',
   'project_id': 287,
   'project_members': [],
   'project_name': 'Radiology_Project_3_mt',
   'resource_id': '5af309c1-4ffb-4aed-a077-d0ed604e8b05',
   'total_tasks': 141},
  {'creation_date': 'Wed, 02 Dec 2020 13:45:24 GMT',
   'group_color': '#fd8c0a',
   'group_id': 9,
   'group_name': 'Radiology_Project',
   'owner': 'mauro',
   'owner_id': '7b6048c8-f923-46e4-9011-2c749e3c2c93',
   'project_description': '',
   'project_id': 285,
   'project_members': [],
   'project_name': 'Radiology_Project_1_cxr',
   'resource_id': '3bae7966-2145-4c6c-ad90-dce1420fd030',
   'total_tasks': 88},
  {'creation_date': 'Fri, 27 Nov 2020 11:54:06 GMT',
   'group_color': '#c58d59',
   'group_id': 8,
   'group_name': 'MIMIC',
   'owner': 'mauro',
   'owner_id': '7b6048c8-f923-46e4-9011-2c749e3c2c93',
   'project_description': '',
   'project_id': 271,
   'project_members': [],
   'project_name': 'MIMIC_Batch1_Andrei',
   'resource_id': 'f0d83c90-ab11-43bb-8ad6-29044d18f492',
   'total_tasks': 100},
  {'creation_date': 'Fri, 27 Nov 2020 11:52:31 GMT',
   'group_color': '#c58d59',
   'group_id': 8,
   'group_name': 'MIMIC',
   'owner': 'mauro',
   'owner_id': '7b6048c8-f923-46e4-9011-2c749e3c2c93',
   'project_description': '',
   'project_id': 270,
   'project_members': [],
   'project_name': 'MIMIC_Batch1_Shalla',
   'resource_id': 'f665dfb7-c8a8-407b-ba04-f9051ec1078d',
   'total_tasks': 100},
  {'creation_date': 'Wed, 25 Nov 2020 16:32:12 GMT',
   'group_color': '#c58d59',
   'group_id': 8,
   'group_name': 'MIMIC',
   'owner': 'mauro',
   'owner_id': '7b6048c8-f923-46e4-9011-2c749e3c2c93',
   'project_description': '',
   'project_id': 157,
   'project_members': [],
   'project_name': 'MIMIC_Batch1_Rebecca',
   'resource_id': 'd3180589-fde5-4497-b156-64f35d623fc2',
   'total_tasks': 100},
  {'creation_date': 'Wed, 25 Nov 2020 16:27:35 GMT',
   'group_color': '#c58d59',
   'group_id': 8,
   'group_name': 'MIMIC',
   'owner': 'mauro',
   'owner_id': '7b6048c8-f923-46e4-9011-2c749e3c2c93',
   'project_description': '',
   'project_id': 156,
   'project_members': [],
   'project_name': 'MIMIC_Batch1',
   'resource_id': 'ae4db43a-ae4f-47b1-ba77-d7d2220892c9',
   'total_tasks': 100},
  {'creation_date': 'Wed, 25 Nov 2020 16:13:20 GMT',
   'group_color': '#c58d59',
   'group_id': 8,
   'group_name': 'MIMIC',
   'owner': 'mauro',
   'owner_id': '7b6048c8-f923-46e4-9011-2c749e3c2c93',
   'project_description': '',
   'project_id': 154,
   'project_members': [],
   'project_name': 'MIMIC_Batch1_Ciprian',
   'resource_id': '71ad67b5-be69-422e-b4e3-25c9b9cf75e6',
   'total_tasks': 100},
  {'creation_date': 'Fri, 06 Nov 2020 12:09:43 GMT',
   'group_color': '#dbdf2e',
   'group_id': 10,
   'group_name': 'MT_Samples',
   'owner': 'mauro',
   'owner_id': '7b6048c8-f923-46e4-9011-2c749e3c2c93',
   'project_description': '',
   'project_id': 127,
   'project_members': [],
   'project_name': 'Internal_Batch5_A_Mauro',
   'resource_id': '2578a602-ba2b-4611-8754-0f687ca280a3',
   'total_tasks': 96},
  {'creation_date': 'Fri, 06 Nov 2020 12:08:02 GMT',
   'group_color': '#dbdf2e',
   'group_id': 10,
   'group_name': 'MT_Samples',
   'owner': 'mauro',
   'owner_id': '7b6048c8-f923-46e4-9011-2c749e3c2c93',
   'project_description': '',
   'project_id': 126,
   'project_members': [],
   'project_name': 'Internal_Batch5_B_Rebecca',
   'resource_id': '7ed36c55-db19-48e0-bc56-4b2114f9a251',
   'total_tasks': 97}],
 'iter_pages': [1, 2],
 'next_num': 2,
 'prev_num': None,
 'total_count': 66}
```

<br />

### Create a new project

```py
alab.create_project(

    # required: unique name of project
    project_name = 'alab_demo',

    # optional: other details about project. Default: Empty string
    project_description='',

    # optional: Sampling option of tasks. Default: random
    project_sampling='',

    # optional: Annotation Guidelines of project
    project_instruction=''
)
```

{:.shell-output}

```sh
Operation completed successfully. Response code: 201
{'project_name': 'alab_demo'}
```

<br />

### Delete a project

```py
alab.delete_project(

    # required: unique name of project
    project_name = 'alab_demo',

    # optional: confirmation for deletion. Default: False - will ask for confirmation. If set to true, will delete directly.
    confirm=False
)
```

{:.shell-output}

```sh
Deleting Project. Press "Y" to confirm.y
Operation completed successfully. Response code: 200
{'message': 'Project successfully Deleted!'}
```

<br />

### Set / Edit configuration of a project

```py
## First, recreate a project
alab.create_project(

    # required: unique name of project
    project_name = 'alab_demo',

    # optional: other details about project. Default: Empty string
    project_description='',

    # optional: Sampling option of tasks. Default: random
    project_sampling='',

    # optional: Annotation Guidelines of project
    project_instruction=''
)
```

```sh
Operation completed successfully. Response code: 201
{'project_name': 'alab_demo'}
```

<br />

#### Set Configuration - First Time

```py
## set configuration - first time
alab.set_project_config(

    # required: name of project
    project_name = 'alab_demo',

    # optional: labels of classes for classification tasks
    classification_labels=['Male', 'Female'],

    # optional: labels of classes for classification tasks
    ner_labels=['Age', 'Symptom', 'Procedure', 'BodyPart'],

    # optional: labels of classes for classification tasks
    assertion_labels=['absent', 'family_history', 'someone_else'],

    # optional: labels of classes for classification tasks
    relations_labels=['is_related']

)
```

{:.shell-output}

```sh
Operation completed successfully. Response code: 201
{'messages': [{'message': 'Project config saved.', 'success': True}]}
```

<br />

#### Edit Configuration - add classes and labels

```py
## Note: At least one type of labels should be provided.
## Note: to define relation labels, NER labels should be provided.

alab.set_project_config(

    # required: name of project
    project_name = 'alab_demo',

    # optional: labels of classes for classification tasks
    classification_labels=['Male', 'Female', 'Unknown'],

    # optional: labels of classes for classification tasks
    ner_labels=['Age', 'Symptom', 'Procedure', 'BodyPart', 'Test', 'Drug'],

    # optional: labels of classes for classification tasks
    assertion_labels=['absent', 'family_history', 'someone_else'],

    # optional: labels of classes for classification tasks
    relations_labels=['is_related', 'is_reactioni_of']

)
```

{:.shell-output}

```sh
Operation completed successfully. Response code: 201
{'messages': [{'message': 'Project config saved.', 'success': True}]}
```

<br />

#### Set Configuration using summary generated at the pre-annotation step

```py
alab.set_project_config(

    # required: name of project
    project_name = 'alab_demo',

    # optional: labels of classes for classification tasks
    classification_labels=['Male', 'Female', 'Unknown'],

    # optional: labels of classes for classification tasks
    ner_labels=summary['ner_labels'],

    # optional: labels of classes for classification tasks
    assertion_labels=summary['assertion_labels'],

    # optional: labels of classes for classification tasks
    relations_labels=['is_related', 'is_reactioni_of']

)
```

{:.shell-output}

```sh
Operation completed successfully. Response code: 201
{'messages': [{'message': 'Project config saved.', 'success': True}]}
```

<br />

### Get configuration of any project

```py
alab.get_project_config(

    # required: name of project
    project_name = 'alab_demo'

)
```

{:.shell-output}

```sh
Operation completed successfully. Response code: 200
{'analytics_permission': {},
 'annotators': ['hasham'],
 'config': {'allow_delete_completions': True,
  'debug': False,
  'editor': {'debug': False},
  'enable_predictions_button': True,
  'input_path': None,
  'instruction': '',
  'ml_backends': [],
  'output_dir': 'completions',
  'port': 8200,
  'sampling': 'uniform',
  'templates_dir': 'examples',
  'title': 'alab_demo'},
 'created_version': '4.0.1',
 'creation_date': 'Thu, 29 Sep 2022 18:46:02 GMT',
 'evaluation_info': None,
 'group_id': None,
 'isVisualNER': None,
 'label_config': '',
 'labels': ['Age',
  'VS_Finding',
  'Gender',
  'Modifier',
  'Duration',
  'RelativeDate',
  'Symptom',
  'Date',
  'External_body_part_or_region',
  'Disease_Syndrome_Disorder',
  'Drug_BrandName',
  'present',
  'absent'],
 'model_types': [{'choices': ['sentiment'], 'name': 'classification'},
  {'name': 'ner'},
  {'name': 'assertion'}],
 'ocr_info': None,
 'owner': {'id': 'ba60df4b-7192-47ca-aa92-759fa577a617', 'username': 'hasham'},
 'project_description': '',
 'project_id': 1130,
 'project_name': 'alab_demo',
 'resource_id': 'e8e17001-a25b-4a92-b419-88948d917647',
 'tasks_count': 0,
 'team_members_order': ['hasham']}
```

<br />

### Upload tasks to a project

```py
# Define a list of tasks/string to upload

txt1 = "The patient is a 21-day-old male here for 2 days of congestion since Nov 8/15 - mom has been suctioning yellow discharge from the patient's nares, plus she has noticed some mild breathing problems while feeding since Nov 9/15 (without signs of perioral cyanosis or retractions). One day ago, mom also noticed a tactile temperature and gave the patient Tylenol."
txt2 = "The patient is a 5-month-old infant who presented initially on Feb 8 with a cold, cough, and runny nose since Feb 2. Mom states she had no fever. She had no difficulty breathing and her cough was described as dry and hacky. At that time, physical exam showed no signs of flu."
task_list = [txt1, txt2]

alab.upload_tasks(

    # required: name of project to upload tasks to
    project_name='alab_demo',

    # required: list of examples / tasks as string (One string is one task).
    task_list=task_list,

    # optional: Option to assign custom titles to tasks. By default, tasks will be titled as 'task_#'
    title_list = [],

    # optional: If there are already tasks in project, then this id offset can be used to make sure default titles 'task_#' do not overlap.
    # While upload a batch after the first one, this can be set to number of tasks currently present in the project
    # This number would be added to each tasks's ID and title.
    id_offset=0

)
```

{:.shell-output}

```sh
Uploading 2 task(s).
Operation completed successfully. Response code: 201
{'completion_count': 0,
 'duration': 0.11868953704833984,
 'failed_count': 0,
 'ignored_count': 0,
 'prediction_count': 0,
 'task_count': 2,
 'task_ids': [1, 2],
 'task_title_warning': 0,
 'updated_count': 0}
```

<br />

### Delete tasks of a project

```py
alab.delete_tasks(

    # required: name of project to upload tasks to
    project_name='alab_demo',

    # required: list of ids of tasks.
    # note: you can get task ids from the above step. Look for 'task_ids' key.
    task_ids=[1, 2],

    # optional: confirmation for deletion. Default: False - will ask for confirmation. If set to true, will delete directly.
    confirm=False
)
```

{:.shell-output}

```sh
Deleting 2 task(s).
Press "Y" to confirm.y
Operation completed successfully. Response code: 200
{'message': 'Task(s) successfully deleted!'}
```

<br />

### Upload pre-annotations to Annotation Lab

You can get the data for `pre_annotations` from [this section](https://nlp.johnsnowlabs.com/docs/en/alab/healthcare#generate-json-containing-pre-annotations-using-a-spark-nlp-pipeline).

```py
alab.upload_preannotations(

    # required: name of project to upload annotations to
    project_name = 'alab_demo',

    # required: preannotation JSON
    preannotations = pre_annotations
    )
```

{:.shell-output}

```sh
Uploading 2 preannotation(s).
Operation completed successfully. Response code: 201
{'completion_count': 0,
 'duration': 0.14992427825927734,
 'failed_count': 0,
 'ignored_count': 0,
 'prediction_count': 2,
 'task_count': 2,
 'task_ids': [1, 2],
 'task_title_warning': 0,
 'updated_count': 0}
```
