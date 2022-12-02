---
layout: model
title: Detect Normalized Genes and Human Phenotypes
author: John Snow Labs
name: ner_human_phenotype_go_clinical
date: 2021-03-31
tags: [ner, clinical, licensed, en]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 3.0.0
spark_version: 3.0
supported: true
annotator: MedicalNerModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model can be used to detect normalized mentions of genes (go) and human phenotypes (hp) in medical text.

## Predicted Entities

: `GO`, `HP`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_HUMAN_PHENOTYPE_GO_CLINICAL/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_human_phenotype_go_clinical_en_3.0.0_3.0_1617209694955.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python

document_assembler = DocumentAssembler()\
	.setInputCol("text")\
	.setOutputCol("document")

sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare", "en", "clinical/models") \
	.setInputCols(["document"]) \
	.setOutputCol("sentence")

tokenizer = Tokenizer()\
	.setInputCols(["sentence"])\
	.setOutputCol("token")
  
word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")

clinical_ner = MedicalNerModel.pretrained("ner_human_phenotype_go_clinical", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner")

ner_converter = NerConverter() \
    .setInputCols(["sentence", "token", "ner"]) \
    .setOutputCol("entities")

nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter])

light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))

annotations = light_pipeline.fullAnnotate("Another disease that shares two of the tumor components of CT, namely GIST and tricarboxylic acid cycle is the Carney-Stratakis syndrome (CSS) or dyad.")
```
```scala

val document_assembler = new DocumentAssembler()
	.setInputCol("text")
	.setOutputCol("document")

val sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare", "en", "clinical/models")
	.setInputCols("document")
	.setOutputCol("sentence")

val tokenizer = new Tokenizer()
	.setInputCols("sentence")
	.setOutputCol("token")

val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("embeddings")

val ner = MedicalNerModel.pretrained("ner_human_phenotype_go_clinical", "en", "clinical/models")
    .setInputCols(Array("sentence", "token", "embeddings"))
    .setOutputCol("ner")

val ner_converter = new NerConverter()
    .setInputCols(Array("sentence", "token", "ner"))
    .setOutputCol("entities")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, ner, ner_converter))

val data = Seq("Another disease that shares two of the tumor components of CT, namely GIST and tricarboxylic acid cycle is the Carney-Stratakis syndrome (CSS) or dyad.").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}

```python

import nlu

nlu.load("en.med_ner.human_phenotype.go_clinical").predict("""Another disease that shares two of the tumor components of CT, namely GIST and tricarboxylic acid cycle is the Carney-Stratakis syndrome (CSS) or dyad.""")
```

</div>

## Results

```bash
+----+--------------------------+---------+-------+----------+
|    | chunk                    |   begin |   end | entity   |
+====+==========================+=========+=======+==========+
|  0 | tumor                    |      39 |    43 | HP       |
+----+--------------------------+---------+-------+----------+
|  1 | tricarboxylic acid cycle |      79 |   102 | GO       |
+----+--------------------------+---------+-------+----------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_human_phenotype_go_clinical|
|Compatibility:|Healthcare NLP 3.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|

## Benchmarking

```bash
|    | label         |    tp |   fp |   fn |     prec |      rec |       f1 |
|---:|--------------:|------:|-----:|-----:|---------:|---------:|---------:|
|  0 | B-GO          | 1530  |  129 |   57 | 0.922242 | 0.964083 | 0.942699 |
|  1 | B-HP          |  950  |  133 |  130 | 0.877193 |  0.87963 |  0.87841 |
|  2 | I-HP          |  253  |   46 |   68 | 0.846154 | 0.788162 | 0.816129 |
|  3 | I-GO          | 4550  |  344 |  154 |  0.92971 | 0.967262 | 0.948114 |
|  4 | Macro-average | 7283  |  652 |  409 | 0.893825 | 0.899784 | 0.896795 |
|  5 | Micro-average | 7283  |  652 |  409 | 0.917832 | 0.946828 | 0.932105 |
```