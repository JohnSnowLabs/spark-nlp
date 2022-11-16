---
layout: model
title: TREC(6) Question Classifier
author: John Snow Labs
name: classifierdl_use_trec6
class: ClassifierDLModel
language: en
repository: public/models
date: 03/05/2020
task: Text Classification
edition: Spark NLP 2.5.0
spark_version: 2.4
tags: [classifier]
supported: true
annotator: ClassifierDLModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description 
Classify open-domain, fact-based questions into one of the following broad semantic categories: Abbreviation, Description, Entities, Human Beings, Locations or Numeric Values.

{:.h2_title}
## Predicted Entities
``ABBR``,  ``DESC``,  ``NUM``,  ``ENTY``,  ``LOC``,  ``HUM``. 

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/CLASSIFICATION_EN_TREC/){:.button.button-orange}<br/>[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/CLASSIFICATION_EN_TREC.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}<br/>[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/classifierdl_use_trec6_en_2.5.0_2.4_1588492648979.zip){:.button.button-orange.button-orange-trans.arr.button-icon}<br/>

## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("document")
use = UniversalSentenceEncoder.pretrained(lang="en") \
.setInputCols(["document"])\
.setOutputCol("sentence_embeddings")
document_classifier = ClassifierDLModel.pretrained('classifierdl_use_trec6', 'en') \
.setInputCols(["document", "sentence_embeddings"]) \
.setOutputCol("class")

nlpPipeline = Pipeline(stages=[documentAssembler, use, document_classifier])
light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))

annotations = light_pipeline.fullAnnotate('When did the construction of stone circles begin in the UK?')

```
```scala
val documentAssembler = DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")
val use = UniversalSentenceEncoder.pretrained(lang="en")
.setInputCols(Array("document"))
.setOutputCol("sentence_embeddings")
val document_classifier = ClassifierDLModel.pretrained("classifierdl_use_trec6", "en")
.setInputCols(Array("document", "sentence_embeddings"))
.setOutputCol("class")
val pipeline = new Pipeline().setStages(Array(documentAssembler, use, document_classifier))

val data = Seq("When did the construction of stone circles begin in the UK?").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu

text = ["""When did the construction of stone circles begin in the UK?"""]
trec6_df = nlu.load('en.classify.trec6.use').predict(text, output_level='document')
trec6_df[["document", "trec6"]]
```

</div>

{:.h2_title}
## Results
{:.table-model}
```bash
+------------------------------------------------------------------------------------------------+------------+
|document                                                                                        |class       |
+------------------------------------------------------------------------------------------------+------------+
|When did the construction of stone circles begin in the UK?                                     | NUM        |
+------------------------------------------------------------------------------------------------+------------+
```

{:.model-param}
## Model Information
{:.table-model}
|-------------------------|--------------------------------------|
| Model Name              | classifierdl_use_trec6               |
| Model Class             | ClassifierDLModel                    |
| Spark Compatibility     | 2.5.0                                |
| Spark NLP Compatibility | 2.4                                  |
| License                 | open source                          |
| Edition                 | public                               |
| Input Labels            |  [document, sentence_embeddings]     |
| Output Labels           | [class]                              |
| Language                | en                                   |
| Upstream Dependencies   | tfhub_use                            |


{:.h2_title}
## Data Source
This model is trained on the 6 class version of TREC dataset. http://search.r-project.org/library/textdata/html/dataset_trec.html

{:.h2_title}
## Benchmarking
```bash
precision    recall  f1-score   support

ABBR       0.00      0.00      0.00        26
DESC       0.89      0.96      0.92       343
ENTY       0.86      0.86      0.86       391
HUM       0.91      0.90      0.91       366
LOC       0.88      0.91      0.89       233
NUM       0.94      0.94      0.94       274

accuracy                           0.89      1633
macro avg       0.75      0.76      0.75      1633
weighted avg       0.88      0.89      0.89      1633
```