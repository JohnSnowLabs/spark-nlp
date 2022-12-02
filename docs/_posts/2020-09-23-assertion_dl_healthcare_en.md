---
layout: model
title: Detect Assertion Status (assertion_dl_healthcare)
author: John Snow Labs
name: assertion_dl_healthcare
class: AssertionDLModel
reference embedding: healthcare_embeddings
language: en
repository: clinical/models
date: 2020-09-23
task: Assertion Status
edition: Healthcare NLP 2.6.0
spark_version: 2.4
tags: [clinical,licensed,assertion,en]
supported: true
annotator: AssertionDLModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Assertion of Clinical Entities based on Deep Learning.  

## Predicted Entities
`hypothetical`, `present`, `absent`, `possible`, `conditional`, `associated_with_someone_else`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/2.Clinical_Assertion_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/assertion_dl_healthcare_en_2.6.0_2.4_1600849811713.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

{:.h2_title}
## How to use 

Use as part of an nlp pipeline with the following stages: DocumentAssembler, SentenceDetector, Tokenizer, WordEmbeddingsModel, NerDLModel, AssertionDLModel.

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
.setInputCols(["sentence", "token"])\
.setOutputCol("embeddings")
clinical_ner = NerDLModel.pretrained("ner_clinical", "en", "clinical/models") \
.setInputCols(["sentence", "token", "embeddings"]) \
.setOutputCol("ner")
ner_converter = NerConverter() \
.setInputCols(["sentence", "token", "ner"]) \
.setOutputCol("ner_chunk")
clinical_assertion = AssertionDLModel.pretrained("assertion_dl_healthcare","en","clinical/models")\
.setInputCols(["document","ner_chunk","embeddings"])\
.setOutputCol("assertion")

nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter, clinical_assertion])

model = nlpPipeline.fit(spark.createDataFrame([['Patient has a headache for the last 2 weeks and appears anxious when she walks fast. No alopecia noted. She denies pain']]).toDF("text"))
results = model.transform(data)
```

```scala
...
val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
.setInputCols(Array("sentence", "token"))
.setOutputCol("embeddings")
val clinical_ner = NerDLModel.pretrained("ner_clinical", "en", "clinical/models")
.setInputCols(Array("sentence", "token", "embeddings")) 
.setOutputCol("ner")
val ner_converter = NerConverter()
.setInputCols(Array("sentence", "token", "ner"))
.setOutputCol("ner_chunk")
val clinical_assertion = AssertionDLModel.pretrained("assertion_dl_healthcare","en","clinical/models")
.setInputCols("document","ner_chunk","embeddings")
.setOutputCol("assertion")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter, clinical_assertion))

val data = Seq("Patient has a headache for the last 2 weeks and appears anxious when she walks fast. No alopecia noted. She denies pain").toDF("text")
val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.assert.healthcare").predict("""Patient has a headache for the last 2 weeks and appears anxious when she walks fast. No alopecia noted. She denies pain""")
```

</div>

{:.h2_title}
## Result
```bash

|   | chunks     | entities| assertion   |
|--:|-----------:|--------:|------------:|
| 0 | a headache | PROBLEM | present     |
| 1 | anxious    | PROBLEM | conditional |
| 2 | alopecia   | PROBLEM | absent      |
| 3 | pain       | PROBLEM | absent      |

```

{:.model-param}
## Model Information

{:.table-model}
|----------------|----------------------------------|
| Name:           | assertion_dl_healthcare          |
| Type:    | AssertionDLModel                 |
| Compatibility:  | 2.6.0                            |
| License:        | Licensed                         |
|Edition:|Official|                       |
|Input labels:         | [document, chunk, word_embeddings] |
|Output labels:        | [assertion]                        |
| Language:       | en                               |
| Case sensitive: | False                            |
| Dependencies:  | embeddings_healthcare_100d       |

{:.h2_title}
## Data Source
Trained using ``embeddings_clinical`` on 2010 i2b2/VA challenge on concepts, assertions, and relations in clinical text from https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/

{:.h2_title}
## Benchmarking
```bash
label  prec    rec     f1

absent  0.9289  0.9466  0.9377
present  0.9433  0.9559  0.9496
conditional  0.6888  0.5     0.5794
associated_with_someone_else  0.9285  0.9122  0.9203
hypothetical  0.9079  0.8654  0.8862
possible  0.7     0.6146  0.6545

macro-avg  0.8496  0.7991  0.8236
micro-avg  0.9245  0.9245  0.9245
```