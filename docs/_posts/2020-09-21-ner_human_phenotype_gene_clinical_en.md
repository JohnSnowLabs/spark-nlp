---
layout: model
title: Detect Genes and Human Phenotypes
author: John Snow Labs
name: ner_human_phenotype_gene_clinical
date: 2020-09-21
task: Named Entity Recognition
language: en
edition: Healthcare NLP 2.6.0
spark_version: 2.4
tags: [ner, en, licensed, clinical]
supported: true
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description
This model detects mentions of genes and human phenotypes (hp) in medical text.
## Predicted Entities
`GENE`, `HP`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_HUMAN_PHENOTYPE_GENE_CLINICAL/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_HUMAN_PHENOTYPE_GENE_CLINICAL.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_human_phenotype_gene_clinical_en_2.5.5_2.4_1598558253840.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_human_phenotype_gene_clinical_en_2.5.5_2.4_1598558253840.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use
Use as part of an nlp pipeline with the following stages: DocumentAssembler, SentenceDetector, Tokenizer, WordEmbeddingsModel, NerDLModel. Add the NerConverter to the end of the pipeline to convert entity tokens into full entity chunks.

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPythonNLU.html %}


```python
...
word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
.setInputCols(["sentence", "token"])\
.setOutputCol("embeddings")
clinical_ner = NerDLModel.pretrained("ner_human_phenotype_gene_clinical", "en", "clinical/models") \
.setInputCols(["sentence", "token", "embeddings"]) \
.setOutputCol("ner")
...
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter])
light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))
annotations = light_pipeline.fullAnnotate("Here we presented a case (BS type) of a 17 years old female presented with polyhydramnios, polyuria, nephrocalcinosis and hypokalemia, which was alleviated after treatment with celecoxib and vitamin D(3).")
```
```scala
...
val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
.setInputCols(Array("sentence", "token"))
.setOutputCol("embeddings")
val ner = NerDLModel.pretrained("ner_human_phenotype_gene_clinical", "en", "clinical/models")
.setInputCols("sentence", "token", "embeddings") 
.setOutputCol("ner")
...
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, ner, ner_converter))
val data = Seq("Here we presented a case (BS type) of a 17 years old female presented with polyhydramnios, polyuria, nephrocalcinosis and hypokalemia, which was alleviated after treatment with celecoxib and vitamin D(3).").toDF("text")
val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.med_ner.human_phenotype.gene_clinical").predict("""Here we presented a case (BS type) of a 17 years old female presented with polyhydramnios, polyuria, nephrocalcinosis and hypokalemia, which was alleviated after treatment with celecoxib and vitamin D(3).""")
```

</div>

{:.h2_title}
## Results
```bash
+----+------------------+---------+-------+----------+
|    | chunk            |   begin |   end | entity   |
+====+==================+=========+=======+==========+
|  0 | BS type          |      29 |    32 | GENE     |
+----+------------------+---------+-------+----------+
|  1 | polyhydramnios   |      75 |    88 | HP       |
+----+------------------+---------+-------+----------+
|  2 | polyuria         |      91 |    98 | HP       |
+----+------------------+---------+-------+----------+
|  3 | nephrocalcinosis |     101 |   116 | HP       |
+----+------------------+---------+-------+----------+
|  4 | hypokalemia      |     122 |   132 | HP       |
+----+------------------+---------+-------+----------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_human_phenotype_gene_clinical|
|Type:|ner|
|Compatibility:|Healthcare NLP 2.6.0 +|
|Edition:|Official|
|License:|Licensed|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|[en]|
|Case sensitive:|false|

## Data source
This model was trained with data from https://github.com/lasigeBioTM/PGR

For further details please refer to https://aclweb.org/anthology/papers/N/N19/N19-1152/

## Benchmarking
```bash
|    | label         |    tp |   fp |   fn |     prec |      rec |       f1 |
|---:|--------------:|------:|-----:|-----:|---------:|---------:|---------:|
|  0 | I-HP          |   303 |   56 |   64 | 0.844011 | 0.825613 | 0.834711 |
|  1 | B-GENE        |  1176 |  158 |  252 | 0.881559 | 0.823529 | 0.851557 |
|  2 | B-HP          |  1078 |  133 |   96 | 0.890173 | 0.918228 | 0.903983 |
|  3 | Macro-average | 2557  | 347  |  412 | 0.871915 | 0.85579  | 0.863777 |
|  4 | Micro-average | 2557  | 347  |  412 | 0.88051  | 0.861233 | 0.870765 |
```
