---
layout: model
title: Aspect based Sentiment Analysis for restaurant reviews
author: John Snow Labs
name: ner_aspect_based_sentiment
date: 2020-12-29
task: Named Entity Recognition
language: en
edition: Spark NLP 2.6.2
spark_version: 2.4
tags: [sentiment, open_source, en, ner]
supported: true
annotator: MedicalNerModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Automatically detect positive, negative and neutral aspects about restaurants from user reviews. Instead of labelling the entire review as negative or positive, this model helps identify which exact phrases relate to sentiment identified in the review.

## Predicted Entities

`NEG`, `POS`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/ASPECT_BASED_SENTIMENT_RESTAURANT/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/ABSA_Inference.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ner_aspect_based_sentiment_en_2.6.2_2.4_1609249232812.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

Use as part of an nlp pipeline with the following stages: DocumentAssembler, SentenceDetector, Tokenizer, WordEmbeddingsModel, NerDLModel. Add the NerConverter to the end of the pipeline to convert entity tokens into full entity chunks.

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
...
word_embeddings = WordEmbeddingsModel.pretrained("glove_6B_300", "xx")\
.setInputCols(["document", "token"])\
.setOutputCol("embeddings")
ner_model = NerDLModel.pretrained("ner_aspect_based_sentiment")\
.setInputCols(["document", "token", "embeddings"])\
.setOutputCol("ner")
...
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, ner_model, ner_converter])
model = nlp_pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))
results = model.transform(spark.createDataFrame([["Came for lunch my sister. We loved our Thai-style main which amazing with lots of flavours very impressive for vegetarian. But the service was below average and the chips were too terrible to finish."]]).toDF("text"))
```
```scala
...
val word_embeddings = WordEmbeddingsModel.pretrained("glove_6B_300", "xx")
.setInputCols(Array("document", "token"))
.setOutputCol("embeddings")
val ner_model = NerDLModel.pretrained("ner_aspect_based_sentiment")
.setInputCols(Array("document", "token", "embeddings"))
.setOutputCol("ner")
...
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, ner_model, ner_converter))
val data = Seq("Came for lunch my sister. We loved our Thai-style main which amazing with lots of flavours very impressive for vegetarian. But the service was below average and the chips were too terrible to finish.").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu
text = ["""Came for lunch my sister. We loved our Thai-style main which amazing with lots of flavours very impressive for vegetarian. But the service was below average and the chips were too terrible to finish."""]

ner_df = nlu.load('en.ner.aspect_sentiment').predict(text, output_level='token')
list(zip(ner_df["entities"].values[0], ner_df["entities_confidence"].values[0])
```

</div>

## Results

```bash
+----------------------------------------------------------------------------------------------------+-------------------+-----------+
| sentence                                                                                           | aspect            | sentiment |
+----------------------------------------------------------------------------------------------------+-------------------+-----------+
| We loved our Thai-style main which amazing with lots of flavours very impressive for vegetarian.   | Thai-style main   | positive  |
| We loved our Thai-style main which amazing with lots of flavours very impressive for vegetarian.   | lots of flavours  | positive  |
| But the service was below average and the chips were too terrible to finish.                       | service           | negative  |
| But the service was below average and the chips were too terrible to finish.                       | chips             | negative  |
+----------------------------------------------------------------------------------------------------+-------------------+-----------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_aspect_based_sentiment|
|Type:|ner|
|Compatibility:|Spark NLP 2.6.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, embeddings]|
|Output Labels:|[absa]|
|Language:|en|
|Dependencies:|glove_6B_300|