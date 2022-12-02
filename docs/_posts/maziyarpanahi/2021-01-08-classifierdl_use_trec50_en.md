---
layout: model
title: TREC(50) Question Classifier
author: John Snow Labs
name: classifierdl_use_trec50
date: 2021-01-08
task: Text Classification
language: en
edition: Spark NLP 2.7.1
spark_version: 2.4
tags: [classifier, text_classification, en, open_source]
supported: true
annotator: ClassifierDLModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Classify open-domain, fact-based questions into sub categories of the following broad semantic categories: Abbreviation, Description, Entities, Human Beings, Locations or Numeric Values.

## Predicted Entities

``ENTY_animal``, ``ENTY_body``, ``ENTY_color``, ``ENTY_cremat``, ``ENTY_currency``, ``ENTY_dismed``, ``ENTY_event``, ``ENTY_food``, ``ENTY_instru``, ``ENTY_lang``, ``ENTY_letter``, ``ENTY_other``, ``ENTY_plant``, ``ENTY_product``, ``ENTY_religion``,  ``ENTY_sport``, ``ENTY_substance``, ``ENTY_symbol``, ``ENTY_techmeth``, ``ENTY_termeq``, ``ENTY_veh``, ``ENTY_word``, ``DESC_def``, ``DESC_desc``, ``DESC_manner``, ``DESC_reason``, ``HUM_gr``, ``HUM_ind``, ``HUM_title``, ``HUM_desc``,  ``LOC_city``, ``LOC_country``, ``LOC_mount``, ``LOC_other``, ``LOC_state``,  ``NUM_code``, ``NUM_count``, ``NUM_date``, ``NUM_dist``, ``NUM_money``, ``NUM_ord``, ``NUM_other``, ``NUM_period``, ``NUM_perc``, ``NUM_speed``, ``NUM_temp``, ``NUM_volsize``, ``NUM_weight``,  ``ABBR_abb``,  ``ABBR_exp``.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/CLASSIFICATION_EN_TREC/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/CLASSIFICATION_EN_TREC.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/classifierdl_use_trec50_en_2.7.1_2.4_1610118328412.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
document_classifier = ClassifierDLModel.pretrained('classifierdl_use_trec50', 'en') \
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
val document_classifier = ClassifierDLModel.pretrained("classifierdl_use_trec50", "en")
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
trec50_df = nlu.load('en.classify.trec50.use').predict(text, output_level = "document")
trec50_df[["document", "trec50"]]
```

</div>

## Results

```bash
+------------------------------------------------------------------------------------------------+------------+
|document                                                                                        |class       |
+------------------------------------------------------------------------------------------------+------------+
|When did the construction of stone circles begin in the UK?                                     | NUM_date   |
+------------------------------------------------------------------------------------------------+------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|classifierdl_use_trec50|
|Compatibility:|Spark NLP 2.7.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|en|

## Data Source

This model is trained on the 50 class version of the TREC dataset. http://search.r-project.org/library/textdata/html/dataset_trec.html
