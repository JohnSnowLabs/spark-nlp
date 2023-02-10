---
layout: model
title: Fake News Classifier in Urdu
author: John Snow Labs
name: classifierdl_urduvec_fakenews
date: 2021-12-29
tags: [urdu, fake_news, fake, ur, open_source]
task: Text Classification
language: ur
edition: Spark NLP 3.3.1
spark_version: 3.0
supported: true
annotator: ClassifierDLModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model determines if news articles in Urdu are real or fake.

## Predicted Entities

`real`, `fake`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/CLASSIFICATION_UR_FAKENEWS/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/CLASSIFICATION_UR_FAKENEWS.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/classifierdl_urduvec_fakenews_ur_3.3.1_3.0_1640771335815.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/classifierdl_urduvec_fakenews_ur_3.3.1_3.0_1640771335815.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler() \
.setInputCol("news") \
.setOutputCol("document")

tokenizer = Tokenizer() \
.setInputCols(["document"]) \
.setOutputCol("token")

normalizer = Normalizer() \
.setInputCols(["token"]) \
.setOutputCol("normalized")

lemma = LemmatizerModel.pretrained("lemma", "ur") \
.setInputCols(["normalized"]) \
.setOutputCol("lemma")

embeddings = WordEmbeddingsModel.pretrained("urduvec_140M_300d", "ur") \
.setInputCols("document", "lemma") \
.setOutputCol("embeddings")

embeddingsSentence = SentenceEmbeddings() \
.setInputCols(["document", "embeddings"]) \
.setOutputCol("sentence_embeddings") \
.setPoolingStrategy("AVERAGE")

classifierdl = ClassifierDLModel.pretrained("classifierdl_urduvec_fakenews", "ur") \
.setInputCols(["document", "sentence_embeddings"]) \
.setOutputCol("class")

urdu_fake_pipeline = Pipeline(stages=[document_assembler, tokenizer, normalizer, lemma, embeddings, embeddingsSentence, classifierdl])

light_pipeline = LightPipeline(urdu_fake_pipeline.fit(spark.createDataFrame([['']]).toDF("news")))

result = light_pipeline.annotate("ایک امریکی تھنک ٹینک نے خبردار کیا ہے کہ جیسے جیسے چین مصنوعی ذہانت (آرٹیفیشل انٹیلی جنس) کے میدان میں ترقی کر رہا ہے، دنیا کا اقتصادی اور عسکری توازن تبدیل ہو سکتا ہے۔")
result["class"]
```
```scala
val document_assembler = DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val tokenizer = Tokenizer()
.setInputCols(Array("document"))
.setOutputCol("token")

val normalizer = Normalizer()
.setInputCols(Array("token"))
.setOutputCol("normalized")

val lemma = LemmatizerModel.pretrained("lemma", "ur") \
.setInputCols(Array("normalized")) \
.setOutputCol("lemma")

val embeddings = WordEmbeddingsModel.pretrained("urduvec_140M_300d", "ur")
.setInputCols(Array("document", "lemma"))
.setOutputCol("embeddings")

val embeddingsSentence = SentenceEmbeddings()
.setInputCols(Array("document", "embeddings"))
.setOutputCol("sentence_embeddings")
.setPoolingStrategy("AVERAGE")

val classifier = ClassifierDLModel.pretrained("classifierdl_urduvec_fakenews", "ur")
.setInputCols(Array("document", "sentence_embeddings"))
.setOutputCol("class")

val urdu_fake_pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, normalizer, lemma, embeddings, embeddingsSentence, classifier))

val light_pipeline = LightPipeline(urdu_fake_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))

val result = light_pipeline.annotate("ایک امریکی تھنک ٹینک نے خبردار کیا ہے کہ جیسے جیسے چین مصنوعی ذہانت (آرٹیفیشل انٹیلی جنس) کے میدان میں ترقی کر رہا ہے، دنیا کا اقتصادی اور عسکری توازن تبدیل ہو سکتا ہے۔")
```


{:.nlu-block}
```python
import nlu
nlu.load("ur.classify.fakenews").predict("""ایک امریکی تھنک ٹینک نے خبردار کیا ہے کہ جیسے جیسے چین مصنوعی ذہانت (آرٹیفیشل انٹیلی جنس) کے میدان میں ترقی کر رہا ہے، دنیا کا اقتصادی اور عسکری توازن تبدیل ہو سکتا ہے۔""")
```

</div>

## Results

```bash
['real']
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|classifierdl_urduvec_fakenews|
|Compatibility:|Spark NLP 3.3.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|ur|
|Size:|21.5 MB|

## Data Source

Combination of multiple open source data sets.

## Benchmarking

```bash
label  precision    recall  f1-score   support
fake       0.77      0.70      0.73       415
real       0.71      0.77      0.74       387
accuracy                           0.73       802
macro-avg       0.74      0.74      0.73       802
weighted-avg       0.74      0.73      0.73       802
```
