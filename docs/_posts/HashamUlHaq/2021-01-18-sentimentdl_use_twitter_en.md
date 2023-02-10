---
layout: model
title: Sentiment Analysis of Tweets (sentimentdl_use_twitter)
author: John Snow Labs
name: sentimentdl_use_twitter
date: 2021-01-18
task: Sentiment Analysis
language: en
edition: Spark NLP 2.7.1
spark_version: 2.4
tags: [en, sentiment, open_source]
supported: true
annotator: SentimentDLModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Classify sentiment in tweets as `negative` or `positive` using `Universal Sentence Encoder` embeddings.

## Predicted Entities

`positive`, `negative`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/SENTIMENT_EN/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/SENTIMENT_EN.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sentimentdl_use_twitter_en_2.7.1_2.4_1610983524713.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sentimentdl_use_twitter_en_2.7.1_2.4_1610983524713.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler() \
.setInputCol("text") \
.setOutputCol("document")

use = UniversalSentenceEncoder.pretrained('tfhub_use', lang="en") \
.setInputCols(["document"])\
.setOutputCol("sentence_embeddings")

classifier = SentimentDLModel().pretrained('sentimentdl_use_twitter')\
.setInputCols(["sentence_embeddings"])\
.setOutputCol("sentiment")

nlp_pipeline = Pipeline(stages=[document_assembler,
use,
classifier
])

l_model = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))

annotations = l_model.fullAnnotate(["im meeting up with one of my besties tonight! Cant wait!!  - GIRL TALK!!", "is upset that he can't update his Facebook by texting it... and might cry as a result  School today also. Blah!"])

```




{:.nlu-block}
```python
import nlu
nlu.load("en.sentiment.twitter.dl").predict("""is upset that he can't update his Facebook by texting it... and might cry as a result  School today also. Blah!""")
```

</div>

## Results

```bash
|    | document                                                                                                         | sentiment   |
|---:|:---------------------------------------------------------------------------------------------------------------- |:------------|
|  0 | im meeting up with one of my besties tonight! Cant wait!!  - GIRL TALK!!                                         | positive    |
|  1 | is upset that he can't update his Facebook by texting it... and might cry as a result  School today also. Blah!  | negative    |

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sentimentdl_use_twitter|
|Compatibility:|Spark NLP 2.7.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[sentiment]|
|Language:|en|
|Dependencies:|tfhub_use|

## Data Source

Trained on Sentiment140 dataset comprising of 1.6M tweets. https://www.kaggle.com/kazanova/sentiment140

## Benchmarking

```bash
loss: 7930.071 - acc: 0.80694044 - val_acc: 80.00508 - batches: 16000
```