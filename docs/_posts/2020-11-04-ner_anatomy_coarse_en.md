---
layout: model
title: Detect Anatomical Structures (Single Entity - embeddings_clinical)
author: John Snow Labs
name: ner_anatomy_coarse_en
date: 2020-11-04
task: Named Entity Recognition
language: en
edition: Healthcare NLP 2.6.1
spark_version: 2.4
tags: [ner, en, licensed, clinical]
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

An NER model to extract all types of anatomical references in text using "embeddings_clinical" embeddings. It is a single entity model and generalizes all anatomical references to a single entity.

## Predicted Entities
`Anatomy`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_ANATOMY/){:.button.button-orange}{:target="_blank"}
[Open in Colab](https://githubtocolab.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}{:target="_blank"}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_anatomy_coarse_en_2.6.1_2.4_1604435935079.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


## How to use

Use as part of an nlp pipeline with the following stages: DocumentAssembler, SentenceDetector, Tokenizer, WordEmbeddingsModel, NerDLModel. Add the NerConverter to the end of the pipeline to convert entity tokens into full entity chunks.

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
...
word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
  .setInputCols(["sentence", "token"])\
  .setOutputCol("embeddings")
clinical_ner = MedicalNerModel.pretrained("ner_anatomy_coarse", "en", "clinical/models") \
  .setInputCols(["sentence", "token", "embeddings"]) \
  .setOutputCol("ner")
...
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter])
model = nlpPipeline.fit(spark.createDataFrame([["content in the lung tissue"]]).toDF("text"))
results = model.transform(data)
```

```scala
...
val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
  .setInputCols(Array("sentence", "token"))
  .setOutputCol("embeddings")
val ner = MedicalNerModel.pretrained("ner_anatomy_coarse", "en", "clinical/models") \
  .setInputCols(["sentence", "token", "embeddings"]) \
  .setOutputCol("ner")
...
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, ner, ner_converter))
val data = Seq("content in the lung tissue").toDF("text")
val result = pipeline.fit(data).transform(data)
```

</div>

{:.h2_title}
## Results
The output is a dataframe with a sentence per row and a ``ner`` column containing all of the entity labels in the sentence, entity character indices, and other metadata. To get only the tokens and entity labels, without the metadata, select ``token.result`` and ``ner.result`` from your output dataframe or add the ``Finisher`` to the end of your pipeline.
```bash
|    | ner_chunk         | entity    |
|---:|------------------:|----------:|
|  0 | lung tissue       | Anatomy   |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_anatomy_coarse|
|Type:|NerDLModel|
|Compatibility:|Spark NLP 2.6.1 +|
|Edition:|Official|
|License:|Licensed|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|[en]|
|Case sensitive:|false|

{:.h2_title}
## Data Source
Trained on a custom dataset using 'embeddings_clinical'.

{:.h2_title}
## Benchmarking
```bash
|    | label         |    tp |    fp |    fn |     prec |      rec |       f1 |
|---:|:--------------|------:|------:|------:|---------:|---------:|---------:|
|  0 | B-Anatomy     |  2568 |   165 |   158 | 0.939627 | 0.94204  | 0.940832 |
|  1 | I-Anatomy     |  1692 |    89 |   169 | 0.950028 | 0.909189 | 0.92916  |
|  2 | Macro-average | 4260  |  254  |   327 | 0.944827 | 0.925614 | 0.935122 |
|  3 | Micro-average | 4260  |  254  |   327 | 0.943731 | 0.928712 | 0.936161 |
```
