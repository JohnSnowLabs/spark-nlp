---
layout: model
title: Spam Classifier
author: John Snow Labs
name: classifierdl_use_spam
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
Automatically identify messages as being regular messages or Spam.

 {:.h2_title}
## Predicted Entities
spam, ham 

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/CLASSIFICATION_EN_SPAM/){:.button.button-orange}<br/>[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/CLASSIFICATION_EN_SPAM.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}<br/>[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/classifierdl_use_spam_en_2.5.3_2.4_1593783318934.zip){:.button.button-orange.button-orange-trans.arr.button-icon}<br/>

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


document_classifier = ClassifierDLModel.pretrained('classifierdl_use_spam', 'en') \
  .setInputCols(["document", "sentence_embeddings"]) \
  .setOutputCol("class")

nlpPipeline = Pipeline(stages=[documentAssembler, use, document_classifier])

light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))

annotations = light_pipeline.fullAnnotate('Congratulations! You've won a $1,000 Walmart gift card. Go to http://bit.ly/1234 to claim now.')

```
```scala
```

</div>

{:.h2_title}
## Results
```bash
+------------------------------------------------------------------------------------------------+------------+
|document                                                                                        |class       |
+------------------------------------------------------------------------------------------------+------------+
|Congratulations! You've won a $1,000 Walmart gift card. Go to http://bit.ly/1234 to claim now.  | spam       |
+------------------------------------------------------------------------------------------------+------------+
```


{:.model-param}
## Model Information

{:.table-model}
|---|---|
| Model Name              | classifierdl_use_spam |
| Model Class             | ClassifierDLModel     |
| Spark Compatibility     | 2.5.3                 |
| Spark NLP Compatibility | 2.4                   |
| License                 | open source           |
| Edition                 | public                |
| Input Labels            | [document, sentence_embeddings] |
| Output Labels           | [class]          |
| Language                | en                    |
| Upstream Dependencies   | tfhub_use             |




{:.h2_title}
## Data Source
This model is trained on UCI spam dataset. https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip

