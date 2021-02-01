---
layout: model
title: Detect Drug Chemicals
author: John Snow Labs
name: ner_drugs_large_en
date: 2021-01-29
tags: [ner, en, licensed, clinical]
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---
{:.h2_title}
## Description

Pretrained named entity recognition deep learning model for Drugs. The model combines dosage, strength, form, and route into a single entity: Drug. The SparkNLP deep learning model (NerDL) is inspired by a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM-CNN. 

{:.h2_title}
## Predicted Entities 
``DRUG``


{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://githubtocolab.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}{:target="_blank"}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_drugs_large_en_2.6.0_2.4_1603915964112.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


## How to use

Use as part of an nlp pipeline with the following stages: DocumentAssembler, SentenceDetector, Tokenizer, WordEmbeddingsModel, NerDLModel. Add the NerConverter to the end of the pipeline to convert entity tokens into full entity chunks.

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}


```python
documentAssembler = DocumentAssembler()\
  .setInputCol("text")\
  .setOutputCol("document")

sentenceDetector = SentenceDetector()\
  .setInputCols(["document"])\
  .setOutputCol("sentence")

tokenizer = Tokenizer()\
  .setInputCols(["sentence"])\
  .setOutputCol("token")

# Clinical word embeddings trained on PubMED dataset
word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
  .setInputCols(["sentence", "token"])\
  .setOutputCol("embeddings")

clinical_ner = NerDLModel.pretrained("ner_drugs_large", "en", "clinical/models") \
  .setInputCols(["sentence", "token", "embeddings"]) \
  .setOutputCol("ner")

ner_converter = NerConverter() \
  .setInputCols(["sentence", "token", "ner"]) \
  .setOutputCol("ner_chunk")

nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter])

model = nlpPipeline.fit(spark.createDataFrame(["The patient is a 40-year-old white male who presents with a chief complaint of "chest pain". The patient is diabetic and has a prior history of coronary artery disease. The patient presents today stating that his chest pain started yesterday evening and has been somewhat intermittent. He has been advised Aspirin 81 milligrams QDay. Humulin N. insulin 50 units in a.m. HCTZ 50 mg QDay. Nitroglycerin 1/150 sublingually PRN chest pain."]).toDF("text"))

results = model.transform(data)

```

```scala
val documentAssembler = DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentenceDetector = SentenceDetector()
  .setInputCols(["document"])
  .setOutputCol("sentence")

val tokenizer = Tokenizer()
  .setInputCols(["sentence"])
  .setOutputCol("token")

# Clinical word embeddings trained on PubMED dataset
val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
  .setInputCols(["sentence", "token"])
  .setOutputCol("embeddings")
  
val ner = NerDLModel.pretrained("ner_drugs_large", "en", "clinical/models")
  .setInputCols("sentence", "token", "embeddings") 
  .setOutputCol("ner")

val ner_converter = NerConverter()
  .setInputCols(["sentence", "token", "ner"])
  .setOutputCol("ner_chunk")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, ner, ner_converter))

val result = pipeline.fit(Seq.empty["The patient is a 40-year-old white male who presents with a chief complaint of "chest pain". The patient is diabetic and has a prior history of coronary artery disease. The patient presents today stating that his chest pain started yesterday evening and has been somewhat intermittent. He has been advised Aspirin 81 milligrams QDay. Humulin N. insulin 50 units in a.m. HCTZ 50 mg QDay. Nitroglycerin 1/150 sublingually PRN chest pain."].toDS.toDF("text")).transform(data)
```

</div>

{:.h2_title}
## Results
The output is a dataframe with a sentence per row and a ``"ner"`` column containing all of the entity labels in the sentence, entity character indices, and other metadata.

```bash

+--------------------------------+---------+
|chunk                           |ner_label|
+--------------------------------+---------+
|Aspirin 81 milligrams           |DRUG     |
|Humulin N                       |DRUG     |
|insulin 50 units                |DRUG     |
|HCTZ 50 mg                      |DRUG     |
|Nitroglycerin 1/150 sublingually|DRUG     |
+--------------------------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_drugs_large_en_2.6.0_2.4|
|Type:|ner|
|Compatibility:|Spark NLP 2.7.1+|
|Edition:|Official|
|License:|Licensed|
|Input Labels:|[sentence,token, embeddings]|
|Output Labels:|[ner]|
|Language:|[en]|
|Case sensitive:|false|

{:.h2_title}
## Data Source
Trained on i2b2_med7 + FDA with 'embeddings_clinical'.
https://www.i2b2.org/NLP/Medication

{:.h2_title}
## Benchmarking

Since this NER model is crafted from `ner_posology` but reduced to single entity, no benchmark is applicable.
