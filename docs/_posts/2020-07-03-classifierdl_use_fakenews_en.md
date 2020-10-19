---
layout: model
title: Fake News Classifier
author: John Snow Labs
name: classifierdl_use_fakenews
class: ClassifierDLModel
language: en
repository: public/models
date: 03/07/2020
tags: [classifier]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description 
Determine if news articles are Real of Fake.

 {:.h2_title}
## Predicted Entities
REAL, FAKE 

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/CLASSIFICATION_EN_FAKENEWS/){:.button.button-orange}<br/>[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/CLASSIFICATION_EN_FAKENEWS.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}<br/>[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/classifierdl_use_fakenews_en_2.5.3_2.4_1593783319296.zip){:.button.button-orange.button-orange-trans.arr.button-icon}<br/>

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


document_classifier = ClassifierDLModel.pretrained('classifierdl_use_fakenews', 'en') \
  .setInputCols(["document", "sentence_embeddings"]) \
  .setOutputCol("class")

nlpPipeline = Pipeline(stages=[documentAssembler, use, document_classifier])

light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))

annotations = light_pipeline.fullAnnotate('Donald Trump a KGB Spy? 11/02/2016 In today’s video, Christopher Greene of AMTV reports Hillary Clinton')

```

</div>

{:.h2_title}
## Results
{:.table-model}
+--------------------------------------------------------------------------------------------------------+------------+
|document                                                                                                |class       |
+--------------------------------------------------------------------------------------------------------+------------+
|Donald Trump a KGB Spy? 11/02/2016 In today’s video, Christopher Greene of AMTV reports Hillary Clinton | FAKE       |
+--------------------------------------------------------------------------------------------------------+------------+


{:.model-param}
## Model Information
{:.table-model}
|-------------------------|---------------------------|
| Model Name              | classifierdl_use_fakenews |
| Model Class             | ClassifierDLModel         |
| Spark Compatibility     | 2.5.3                     |
| Spark NLP Compatibility | 2.4                       |
| License                 | open source               |
| Edition                 | public                    |
| Input Labels            | [document, sentence_embeddings] |
| Output Labels           | [class]               |
| Language                | en                        |
| Upstream Dependencies   | with tfhub_use            |




{:.h2_title}
## Data Source
This model is trained on the fake new classification challenge. https://raw.githubusercontent.com/joolsa/fake_real_news_dataset/master/fake_or_real_news.csv.zip

