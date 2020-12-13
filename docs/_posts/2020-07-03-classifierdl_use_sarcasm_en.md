---
layout: model
title: Sarcasm Classifier
author: John Snow Labs
name: classifierdl_use_sarcasm
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
Classify if a text contains sarcasm.

 {:.h2_title}
## Predicted Entities
normal, sarcasm 

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/SENTIMENT_EN_SARCASM/){:.button.button-orange}<br/>[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/SENTIMENT_EN_SARCASM.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}<br/>[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/classifierdl_use_sarcasm_en_2.5.3_2.4_1593783319298.zip){:.button.button-orange.button-orange-trans.arr.button-icon}<br/>

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


document_classifier = ClassifierDLModel.pretrained('classifierdl_use_sarcasm', 'en') \
  .setInputCols(["document", "sentence_embeddings"]) \
  .setOutputCol("class")

nlpPipeline = Pipeline(stages=[documentAssembler, use, document_classifier])

light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))

annotations = light_pipeline.fullAnnotate('If I could put into words how much I love waking up at am on Tuesdays I would')

```

</div>

{:.h2_title}
## Results
{:.table-model}
+--------------------------------------------------------------------------------------------------------+------------+
|document                                                                                                |class       |
+--------------------------------------------------------------------------------------------------------+------------+
|If I could put into words how much I love waking up at am on Tuesdays I would                           | sarcasm    |
+--------------------------------------------------------------------------------------------------------+------------+


{:.model-param}
## Model Information
{:.table-model}
|-------------------------|--------------------------|
| Model Name              | classifierdl_use_sarcasm |
| Model Class             | ClassifierDLModel        |
| Spark Compatibility     | 2.5.3                    |
| Spark NLP Compatibility | 2.4                      |
| License                 | open source              |
| Edition                 | public                   |
| Input Labels            | [document, sentence_embeddings] |
| Output Labels           | [class]         |
| Language                | en                       |
| Upstream Dependencies   | with tfhub_use           |




{:.h2_title}
## Data Source
This model is trained on the sarcam detection dataset. https://github.com/MirunaPislar/Sarcasm-Detection/tree/master/res/datasets/riloff

