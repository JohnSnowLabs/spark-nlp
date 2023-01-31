---
layout: model
title: Detect biological concepts (biobert)
author: John Snow Labs
name: ner_bionlp_biobert
date: 2021-04-01
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

Detect general biological entities like tissues, organisms, cells, etc in text using pretrained NER model.

## Predicted Entities

`tissue_structure`, `Amino_acid`, `Simple_chemical`, `Organism_substance`, `Developing_anatomical_structure`, `Cell`, `Cancer`, `Cellular_component`, `Gene_or_gene_product`, `Immaterial_anatomical_entity`, `Organ`, `Organism`, `Pathological_formation`, `Organism_subdivision`, `Anatomical_system`, `Tissue`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_TUMOR/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_bionlp_biobert_en_3.0.0_3.0_1617260864949.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_bionlp_biobert_en_3.0.0_3.0_1617260864949.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

embeddings_clinical = BertEmbeddings.pretrained("biobert_pubmed_base_cased")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")

clinical_ner = MedicalNerModel.pretrained("ner_bionlp_biobert", "en", "clinical/models")\
    .setInputCols(["sentence", "token", "embeddings"])\
    .setOutputCol("ner")

ner_converter = NerConverter()\
 	.setInputCols(["sentence", "token", "ner"])\
 	.setOutputCol("ner_chunk")
    
nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings_clinical, clinical_ner, ner_converter])

model = nlpPipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

results = model.transform(spark.createDataFrame([["EXAMPLE_TEXT"]]).toDF("text"))
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

val embeddings_clinical = BertEmbeddings.pretrained("biobert_pubmed_base_cased")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("embeddings")

val ner = MedicalNerModel.pretrained("ner_bionlp_biobert", "en", "clinical/models")
    .setInputCols(Array("sentence", "token", "embeddings"))
    .setOutputCol("ner")

val ner_converter = new NerConverter()
 	.setInputCols(Array("sentence", "token", "ner"))
 	.setOutputCol("ner_chunk")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings_clinical, ner, ner_converter))

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.med_ner.bionlp.biobert").predict("""Put your text here.""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_bionlp_biobert|
|Compatibility:|Healthcare NLP 3.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|


## Benchmarking
```bash
+-------------------------------+------+-----+------+------+---------+------+------+
|                         entity|    tp|   fp|    fn| total|precision|recall|    f1|
+-------------------------------+------+-----+------+------+---------+------+------+
|                          Organ| 123.0| 59.0|  50.0| 173.0|   0.6758| 0.711| 0.693|
|         Pathological_formation|  82.0| 32.0|  45.0| 127.0|   0.7193|0.6457|0.6805|
|             Organism_substance|  75.0|  8.0|  51.0| 126.0|   0.9036|0.5952|0.7177|
|               tissue_structure| 412.0|162.0|  53.0| 465.0|   0.7178| 0.886|0.7931|
|             Cellular_component| 188.0| 46.0|  61.0| 249.0|   0.8034| 0.755|0.7785|
|                         Tissue| 244.0| 73.0|  51.0| 295.0|   0.7697|0.8271|0.7974|
|                         Cancer|1384.0|193.0| 144.0|1528.0|   0.8776|0.9058|0.8915|
|Developing_anatomical_structure|  10.0|  3.0|  11.0|  21.0|   0.7692|0.4762|0.5882|
|   Immaterial_anatomical_entity|  20.0| 16.0|  21.0|  41.0|   0.5556|0.4878|0.5195|
|           Gene_or_gene_product|3829.0|233.0|1045.0|4874.0|   0.9426|0.7856| 0.857|
|                           Cell|1873.0|175.0| 231.0|2104.0|   0.9146|0.8902|0.9022|
|                       Organism| 559.0|116.0|  79.0| 638.0|   0.8281|0.8762|0.8515|
|                Simple_chemical| 928.0|106.0| 421.0|1349.0|   0.8975|0.6879|0.7789|
+-------------------------------+------+-----+------+------+---------+------+------+

+------------------+
|             macro|
+------------------+
|0.6589490623994527|
+------------------+

+------------------+
|             micro|
+------------------+
|0.8407823737981155|
+------------------+
```