---
layout: model
title: Detect Problems, Tests and Treatments (ner_healthcare)
author: John Snow Labs
name: ner_healthcare_en
date: 2020-03-26
task: Named Entity Recognition
language: en
edition: Healthcare NLP 2.4.4
spark_version: 2.4
tags: [ner, en, licensed, clinical]
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Pretrained named entity recognition deep learning model for healthcare. Includes Problem, Test and Treatment entities. The SparkNLP deep learning model (NerDL) is inspired by a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM-CNN. 

{:.h2_title}
## Predicted Entities 
``PROBLEM``, ``TEST``, ``TREATMENT``.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_CLINICAL/){:.button.button-orange}
[Open in Colab](https://githubtocolab.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}{:target="_blank"}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_healthcare_en_2.4.4_2.4_1585188313964.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

{:.h2_title}
## How to use

Use as part of an nlp pipeline with the following stages: DocumentAssembler, SentenceDetector, Tokenizer, WordEmbeddingsModel, NerDLModel. Add the NerConverter to the end of the pipeline to convert entity tokens into full entity chunks.

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}


```python
...
word_embeddings = WordEmbeddingsModel.pretrained("embeddings_healthcare_100d", "en", "clinical/models")\
  .setInputCols(["sentence", "token"])\
  .setOutputCol("embeddings")
clinical_ner = NerDLModel.pretrained("ner_healthcare", "en", "clinical/models") \
  .setInputCols(["sentence", "token", "embeddings"]) \
  .setOutputCol("ner")
...
nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter])

model = nlpPipeline.fit(spark.createDataFrame([["A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus ( T2DM ), one prior episode of HTG-induced pancreatitis three years prior to presentation , associated with an acute hepatitis , and obesity with a body mass index ( BMI ) of 33.5 kg/m2 , presented with a one-week history of polyuria , polydipsia , poor appetite , and vomiting . Two weeks prior to presentation , she was treated with a five-day course of amoxicillin for a respiratory tract infection . She was on metformin , glipizide , and dapagliflozin for T2DM and atorvastatin and gemfibrozil for HTG ."]]).toDF("text"))

results = model.transform(data)

```

```scala
...
val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_healthcare_100d", "en", "clinical/models")
  .setInputCols(Array("sentence", "token"))
  .setOutputCol("embeddings")
val ner = NerDLModel.pretrained("ner_healthcare", "en", "clinical/models") 
  .setInputCols(Array("sentence", "token", "embeddings"))
  .setOutputCol("ner")
...
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, ner, ner_converter))

val data = Seq(A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus ( T2DM ), one prior episode of HTG-induced pancreatitis three years prior to presentation , associated with an acute hepatitis , and obesity with a body mass index ( BMI ) of 33.5 kg/m2 , presented with a one-week history of polyuria , polydipsia , poor appetite , and vomiting . Two weeks prior to presentation , she was treated with a five-day course of amoxicillin for a respiratory tract infection . She was on metformin , glipizide , and dapagliflozin for T2DM and atorvastatin and gemfibrozil for HTG .).toDF("text")
val result = pipeline.fit(data).transform(data)

```

</div>

{:.h2_title}
## Results
The output is a dataframe with a sentence per row and a ``"ner"`` column containing all of the entity labels in the sentence, entity character indices, and other metadata. To get only the tokens and entity labels, without the metadata, select ``"token.result"`` and ``"ner.result"`` from your output dataframe or add the ``"Finisher"`` to the end of your pipeline.
```bash
|   | chunk                         | ner_label |
|---|-------------------------------|-----------|
| 0 | a respiratory tract infection | PROBLEM   |
| 1 | metformin                     | TREATMENT |
| 2 | glipizide                     | TREATMENT |
| 3 | dapagliflozin                 | TREATMENT |
| 4 | T2DM                          | PROBLEM   |
| 5 | atorvastatin                  | TREATMENT |
| 6 | gemfibrozil                   | TREATMENT |

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_healthcare_en_2.4.4_2.4|
|Type:|ner|
|Compatibility:|Spark NLP 2.4.4|
|Edition:|Official|
|License:|Licensed|
|Input Labels:|[sentence,token, embeddings]|
|Output Labels:|[ner]|
|Language:|[en]|
|Case sensitive:|false|

{:.h2_title}
## Data Source
Trained on 2010 i2b2 challenge data with 'embeddings_healthcare_100d'.
https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/

{:.h2_title}
## Benchmarking
```bash
|    | label         |    tp |    fp |    fn |     prec |      rec |       f1 |
|---:|:--------------|------:|------:|------:|---------:|---------:|---------:|
|  0 | I-TREATMENT   |  6625 |  1187 |  1329 | 0.848054 | 0.832914 | 0.840416 |
|  1 | I-PROBLEM     | 15142 |  1976 |  2542 | 0.884566 | 0.856254 | 0.87018  |
|  2 | B-PROBLEM     | 11005 |  1065 |  1587 | 0.911765 | 0.873968 | 0.892466 |
|  3 | I-TEST        |  6748 |   923 |  1264 | 0.879677 | 0.842237 | 0.86055  |
|  4 | B-TEST        |  8196 |   942 |  1029 | 0.896914 | 0.888455 | 0.892665 |
|  5 | B-TREATMENT   |  8271 |  1265 |  1073 | 0.867345 | 0.885167 | 0.876165 |
|  6 | Macro-average | 55987 |  7358 |  8824 | 0.881387 | 0.863166 | 0.872181 |
|  7 | Micro-average | 55987 |  7358 |  8824 | 0.883842 | 0.86385  | 0.873732 |
```