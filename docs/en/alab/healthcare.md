---
layout: docs
comment: no
header: true
seotitle: Annotation Lab | John Snow Labs
title: NLP Libraries Integration
permalink: /docs/en/alab/healthcare
key: docs-training
modify_date: "2022-11-23"
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

 <a href="/docs/en/licensed_install">Healthcare NLP</a> provides an easy to use module for interacting with Annotation Lab with minimal code. In this section, you can find the instructions for performing specific operations using the annotation lab module of the <a href="/docs/en/licensed_install">Healthcare NLP</a> library. You can execute these instructions in a python notebook (Jupyter, Colab, Kaggle, etc.).

Before running the instructions described in the following sub-sections, some initial environment setup needs to be performed in order to configure the Healthcare NLP library and start a Spark session.

> **NOTE:** For using this integration a Healthcare, Finance and/or Legal NLP License key is requirend. If you do not have one, you can get it <a href="https://www.johnsnowlabs.com/install/"> here</a>.  

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

> **NOTE:** The license upload widget is only available when the cell has been executed in the current browser session. Please rerun this cell to enable.

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
{:.info}
**Using already exported JSON to generate training data - _No Annotation Lab Credentials Required_**

```py
# import the module
from sparknlp_jsl.alab import AnnotationLab
alab = AnnotationLab()
```
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
**_No Annotation Lab Credentials Required_. Only the exported JSON is used.**

### Classification Model

The following snippet shows how to generate data for training a classification model.

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

The JSON export must be converted into a CoNLL format suitable for training an NER model.

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

The JSON export is converted into a dataframe, suitable for training an assertion model.

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

The JSON export is converted into a dataframe suitable for training a relation extraction model.

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

## Generate Pre-annotations using Spark NLP pipelines

{:.info}
**No Annotation Lab credentials are required.**

The first step is to define the Healthcare NLP pipeline. The same procedure can be followed for Legal and Finance NLP pipelines. 

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

**Generate pre-annotation JSON using pipeline results**

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

The Generated JSON can be uploaded to Annotation Lab to particular project directly via UI or via [API](https://nlp.johnsnowlabs.com/docs/en/alab/healthcare#upload-pre-annotations-to-annotation-lab).

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

An annotation summary is also generated that can be used to [setup and configure a new project](/docs/en/alab/healthcare#set-configuration-using-summary-generated-at-the-pre-annotation-step).

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
**Credentials are required for the following actions.**

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

    # required: secret for you Annotation Lab instance (every Annotation Lab installation has a different secret)
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
{'has_next': False,
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
  {'creation_date': 'Fri, 06 Nov 2020 12:08:02 GMT',
   'group_color': '#dbdf2e',
   'group_id': 10,
   'group_name': 'MT_Samples',
   'owner': 'mauro',
   'owner_id': '7b6048c8-f923-46e4-9011-2c749e3c2c93',
   'project_description': '',
   'project_id': 126,
   'project_members': [],
   'project_name': 'PathologyReports',
   'resource_id': '7ed36c55-db19-48e0-bc56-4b2114f9a251',
   'total_tasks': 97}],
 'iter_pages': [1],
 'next_num': None,
 'prev_num': None,
 'total_count': 3}
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
