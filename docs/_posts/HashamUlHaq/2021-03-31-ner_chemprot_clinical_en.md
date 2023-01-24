---
layout: model
title: Detect Chemical Compounds and Genes
author: John Snow Labs
name: ner_chemprot_clinical
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

This is a pre-trained model that can be used to automatically detect all chemical compounds and gene mentions from medical texts.

## Predicted Entities

: `CHEMICAL`, `GENE-Y`, `GENE-N`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_CHEMPROT_CLINICAL/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_CHEMPROT_CLINICAL.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_chemprot_clinical_en_3.0.0_3.0_1617208430062.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_chemprot_clinical_en_3.0.0_3.0_1617208430062.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")
         
sentence_detector = SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")

clinical_ner = MedicalNerModel.pretrained("ner_chemprot_clinical", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner")

ner_converter = NerConverter()\
 	  .setInputCols(["sentence", "token", "ner"])\
 	  .setOutputCol("ner_chunk")

nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter])

model = nlpPipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

results = model.transform(spark.createDataFrame([["Keratinocyte growth factor and acidic fibroblast growth factor are mitogens for primary cultures of mammary epithelium."]]).toDF("text"))
```
```scala
val document_assembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")
         
val sentence_detector = new SentenceDetector()
    .setInputCols("document")
    .setOutputCol("sentence")

val tokenizer = new Tokenizer()
    .setInputCols("sentence")
    .setOutputCol("token")

val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("embeddings")

val ner = MedicalNerModel.pretrained("ner_chemprot_clinical", "en", "clinical/models")
    .setInputCols(Array("sentence", "token", "embeddings"))
    .setOutputCol("ner")

val ner_converter = new NerConverter()
 	.setInputCols(Array("sentence", "token", "ner"))
 	.setOutputCol("ner_chunk")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, ner, ner_converter))

val data = Seq("""Keratinocyte growth factor and acidic fibroblast growth factor are mitogens for primary cultures of mammary epithelium.""").toDS().toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.med_ner.chemprot.clinical").predict("""Keratinocyte growth factor and acidic fibroblast growth factor are mitogens for primary cultures of mammary epithelium.""")
```

</div>

## Results

```bash
+----+---------------------------------+---------+-------+----------+
|    | chunk                           |   begin |   end | entity   |
+====+=================================+=========+=======+==========+
|  0 | Keratinocyte growth factor      |       0 |    25 | GENE-Y   |
+----+---------------------------------+---------+-------+----------+
|  1 | acidic fibroblast growth factor |      31 |    61 | GENE-Y   |
+----+---------------------------------+---------+-------+----------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_chemprot_clinical|
|Compatibility:|Healthcare NLP 3.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|

## Data Source

This model was trained on the <a href="https://biocreative.bioinformatics.udel.edu/"> ChemProt corpus</a> using 'embeddings_clinical' embeddings. Make sure you use the same embeddings when running the model.

## Benchmarking

```bash
|    | label         |     tp |    fp |   fn |     prec |      rec |       f1 |
|---:|:--------------|-------:|------:|-----:|---------:|---------:|---------:|
|  0 | B-GENE-Y      |   4650 |  1090 |  838 | 0.810105 | 0.847303 | 0.828286 |
|  1 | B-GENE-N      |   1732 |   981 | 1019 | 0.638408 | 0.629589 | 0.633968 |
|  2 | I-GENE-Y      |   1846 |   571 |  573 | 0.763757 | 0.763125 | 0.763441 |
|  3 | B-CHEMICAL    |   7512 |   804 | 1136 | 0.903319 | 0.86864  | 0.88564  |
|  4 | I-CHEMICAL    |   1059 |   169 |  253 | 0.862378 | 0.807165 | 0.833858 |
|  5 | I-GENE-N      |   1393 |   853 |  598 | 0.620214 | 0.699648 | 0.657541 |
|  6 | Macro-average | 18192  | 4468  | 4417 | 0.766363 | 0.769245 | 0.767801 |
|  7 | Micro-average | 18192  | 4468  | 4417 | 0.802824 | 0.804635 | 0.803729 |
```