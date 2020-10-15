---
layout: model
title: TREC(50) Question Classifier
author: John Snow Labs
name: classifierdl_use_trec6
class: ClassifierDLModel
language: en
repository: public/models
date: 03/05/2020
tags: [classifier]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description 
Classify open-domain, fact-based questions into one of the following broad semantic categories: Abbreviation, Description, Entities, Human Beings, Locations or Numeric Values.

 {:.h2_title}
## Predicted Entities
 ABBR,  DESC,  NUM,  ENTY,  LOC,  HUM 

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/CLASSIFICATION_EN_TREC/){:.button.button-orange}<br/>[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/CLASSIFICATION_EN_TREC.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}<br/>[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/classifierdl_use_trec6_en_2.5.0_2.4_1588492648979.zip){:.button.button-orange.button-orange-trans.arr.button-icon}<br/>

## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

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

nlpPipeline = Pipeline(stages=[
                               documentAssembler, 
                               use,
                               document_classifier
                               ])

light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))

annotations = light_pipeline.fullAnnotate('When did the construction of stone circles begin in the UK?')

```

</div>

{:.h2_title}
## Results
{:.table-model}
+------------------------------------------------------------------------------------------------+------------+
|document                                                                                        |class       |
+------------------------------------------------------------------------------------------------+------------+
|When did the construction of stone circles begin in the UK?                                     | NUM        |
+------------------------------------------------------------------------------------------------+------------+

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
| Input Labels            |                                      |
| Output Labels           | ABBR,  DESC,  NUM,  ENTY,  LOC,  HUM |
| Language                | en                                   |
| Dimension               |                                      |
| Case Sensitive          |                                      |
| Upstream Dependencies   | tfhub_use                            |




{:.h2_title}
## Data Source
This model is trained on the 6 class version of TREC dataset. http://search.r-project.org/library/textdata/html/dataset_trec.html

