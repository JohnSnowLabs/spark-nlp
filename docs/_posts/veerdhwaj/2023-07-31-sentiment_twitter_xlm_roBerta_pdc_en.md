---
layout: model
title: Sentiment tags predictions trained on twitter
author: veerdhwaj
name: sentiment_twitter_xlm_roBerta_pdc
date: 2023-07-31
tags: [sentiment, en, open_source, tensorflow]
task: Text Classification
language: en
edition: Spark NLP 3.3.2
spark_version: 3.2
supported: false
engine: tensorflow
annotator: XlmRoBertaForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a multilingual XLM-roBERTa-base model trained on ~198M tweets and finetuned for sentiment analysis. The sentiment fine-tuning was done on 8 languages (Ar, En, Fr, De, Hi, It, Sp, Pt) but it can be used for more languages (see paper for details).

Paper: XLM-T: A Multilingual Language Model Toolkit for Twitter.
Git Repo: XLM-T official repository.
This model has been integrated into the TweetNLP library.

HF Model: https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment

## Predicted Entities

`class`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/community.johnsnowlabs.com/veerdhwaj/sentiment_twitter_xlm_roBerta_pdc_en_3.3.2_3.2_1690780495131.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://community.johnsnowlabs.com/veerdhwaj/sentiment_twitter_xlm_roBerta_pdc_en_3.3.2_3.2_1690780495131.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from pyspark.ml import Pipeline
from sparknlp.annotator import *

document_assembler = DocumentAssembler() \
    .setInputCol('text') \
    .setOutputCol('document')

tokenizer = Tokenizer() \
    .setInputCols(['document']) \
    .setOutputCol('token')

sequenceClassifier = XlmRoBertaForSequenceClassification.pretrained('sentiment_twitter_xlm_roBerta_pdc_en')\
  .setInputCols(["document",'token'])\
  .setOutputCol("class")

pipeline = Pipeline(stages=[
    document_assembler, 
    tokenizer,
    sequenceClassifier    
])

# couple of simple examples
example = spark.createDataFrame([['사랑해!'], ["T'estimo! ❤️"], ["I love you!"], ['Mahal kita!']]).toDF("text")

result = pipeline.fit(example).transform(example)

# result is a DataFrame
result.select("text", "class.result").show()
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sentiment_twitter_xlm_roBerta_pdc|
|Compatibility:|Spark NLP 3.3.2+|
|License:|Open Source|
|Edition:|Community|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|1.0 GB|
|Case sensitive:|true|
|Max sentence length:|512|
|Dependencies:|xlm_roBerta|