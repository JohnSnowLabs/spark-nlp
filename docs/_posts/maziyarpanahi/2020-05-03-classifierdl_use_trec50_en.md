---
layout: model
title: TREC(50) Question Classifier
author: John Snow Labs
name: classifierdl_use_trec50
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
Classify open-domain, fact-based questions into sub categories of the following broad semantic categories: Abbreviation, Description, Entities, Human Beings, Locations or Numeric Values.

{:.h2_title}
## Predicted Entities
``ENTY_animal``, ``ENTY_body``, ``ENTY_color``, ``ENTY_cremat``, ``ENTY_currency``, ``ENTY_dismed``, ``ENTY_event``, ``ENTY_food``, ``ENTY_instru``, ``ENTY_lang``, ``ENTY_letter``, ``ENTY_other``, ``ENTY_plant``, ``ENTY_product``, ``ENTY_religion``,  ``ENTY_sport``, ``ENTY_substance``, ``ENTY_symbol``, ``ENTY_techmeth``, ``ENTY_termeq``, ``ENTY_veh``, ``ENTY_word``, ``DESC_def``, ``DESC_desc``, ``DESC_manner``, ``DESC_reason``, ``HUM_gr``, ``HUM_ind``, ``HUM_title``, ``HUM_desc``,  ``LOC_city``, ``LOC_country``, ``LOC_mount``, ``LOC_other``, ``LOC_state``,  ``NUM_code``, ``NUM_count``, ``NUM_date``, ``NUM_dist``, ``NUM_money``, ``NUM_ord``, ``NUM_other``, ``NUM_period``, ``NUM_perc``, ``NUM_speed``, ``NUM_temp``, ``NUM_volsize``, ``NUM_weight``,  ``ABBR_abb``,  ``ABBR_exp``. 

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/CLASSIFICATION_EN_TREC/){:.button.button-orange}<br/>[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/CLASSIFICATION_EN_TREC.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}<br/>[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/classifierdl_use_trec50_en_2.5.0_2.4_1588493558481.zip){:.button.button-orange.button-orange-trans.arr.button-icon}<br/>

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
text = ["""William Henry Gates III (born October 28, 1955) is an American business magnate, software developer, investor, and philanthropist. He is best known as the co-founder of Microsoft Corporation. During his career at Microsoft, Gates held the positions of chairman, chief executive officer (CEO), president and chief software architect, while also being the largest individual shareholder until May 2014. He is one of the best-known entrepreneurs and pioneers of the microcomputer revolution of the 1970s and 1980s. Born and raised in Seattle, Washington, Gates co-founded Microsoft with childhood friend Paul Allen in 1975, in Albuquerque, New Mexico; it went on to become the world's largest personal computer software company. Gates led the company as chairman and CEO until stepping down as CEO in January 2000, but he remained chairman and became chief software architect. During the late 1990s, Gates had been criticized for his business tactics, which have been considered anti-competitive. This opinion has been upheld by numerous court rulings. In June 2006, Gates announced that he would be transitioning to a part-time role at Microsoft and full-time work at the Bill & Melinda Gates Foundation, the private charitable foundation that he and his wife, Melinda Gates, established in 2000. He gradually transferred his duties to Ray Ozzie and Craig Mundie. He stepped down as chairman of Microsoft in February 2014 and assumed a new post as technology adviser to support the newly appointed CEO Satya Nadella."""]

trec50_df = nlu.load('en.classify.trec50.use').predict(text, output_level = "document")
trec50_df[["document", "trec50"]]
```

</div>


{:.h2_title}
## Results
{:.table-model}
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

| Model Name              | classifierdl_use_trec50  |
| Model Class             | ClassifierDLModel       |
| Spark Compatibility     | 2.5.0 |
| Spark NLP Compatibility | 2.4 |
| License                 | open source|
| Edition                 | public |
| Input Labels            |  [document, sentence_embeddings]     |
| Output Labels           | [class]                              |
| Language                | en|
| Upstream Dependencies   | with tfhub_use |


{:.h2_title}
## Data Source
This model is trained on the 50 class version of TREC dataset. http://search.r-project.org/library/textdata/html/dataset_trec.html
