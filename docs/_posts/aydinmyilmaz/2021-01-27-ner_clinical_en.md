---
layout: model
title: Detect Problem, Test and Treatment Entities  (ner_clinical)
author: John Snow Labs
name: ner_clinical
date: 2021-01-27
task: Named Entity Recognition
language: en
edition: Spark NLP for Healthcare 2.7.2
spark_version: 2.4
tags: [en, clinical, ner, licensed]
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained named entity recognition deep learning model for clinical terms. The SparkNLP deep learning model (NerDL) is inspired by a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM-CNN.

## Predicted Entities

`Problem`, `Test`, `Treatment`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_CLINICAL/){:.button.button-orange}
[Open in Colab](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_clinical_en_2.7.2_2.4_1611751979087.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

Use as part of an nlp pipeline with the following stages: DocumentAssembler, SentenceDetector, Tokenizer, WordEmbeddingsModel, NerDLModel. Add the NerConverter to the end of the pipeline to convert entity tokens into full entity chunks.

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
...
clinical_ner = NerDLModel.pretrained("ner_clinical_large", "en", "clinical/models") \
  .setInputCols(["sentence", "token", "embeddings"]) \
  .setOutputCol("ner")
...

nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings_clinical, clinical_ner, ner_converter])

model = nlpPipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

results = model.transform(data)
```

```scala
...
val clinical_ner = NerDLModel.pretrained("ner_clinical_large", "en", "clinical/models")
  .setInputCols(Array("sentence", "token", "embeddings"))
  .setOutputCol("ner")

val nlpPipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings_clinical, clinical_ner, ner_converter))

val result = pipeline.fit(Seq.empty[String]).transform(data)

val results = LightPipeline(model).fullAnnotate(data)
```



{:.nlu-block}
```python
import nlu
nlu.load("en.med_ner.clinical").predict("""Put your text here.""")
```

</div>

## Results

```bash
The output is a dataframe with a sentence per row and a `ner` column containing all of the entity labels in the sentence, entity character indices, and other metadata. To get only the tokens and entity labels, without the metadata, select `token.result` and `ner.result` from your output dataframe:  
  


+-----------------------------------------------------------+---------+
|chunk                                                      |ner_label|
+-----------------------------------------------------------+---------+
|the G-protein-activated inwardly rectifying potassium (GIRK|TREATMENT|
|the genomicorganization                                    |TREATMENT|
|a candidate gene forType II diabetes mellitus              |PROBLEM  |
|byapproximately                                            |TREATMENT|
|single nucleotide polymorphisms                            |TREATMENT|
|aVal366Ala substitution                                    |TREATMENT|
|an 8 base-pair                                             |TREATMENT|
|insertion/deletion                                         |PROBLEM  |
|Ourexpression studies                                      |TEST     |
|the transcript in various humantissues                     |PROBLEM  |
|fat andskeletal muscle                                     |PROBLEM  |
|furtherstudies                                             |PROBLEM  |
|the KCNJ9 protein                                          |TREATMENT|
|evaluation                                                 |TEST     |
|Type II diabetes                                           |PROBLEM  |
|the treatment                                              |TREATMENT|
|breast cancer                                              |PROBLEM  |
|the standard therapy                                       |TREATMENT|
|anthracyclines                                             |TREATMENT|
|taxanes                                                    |TREATMENT|
+-----------------------------------------------------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_clinical|
|Type:|ner|
|Compatibility:|Spark NLP 2.7.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Dependencies:|embeddings_clinical|

## Data Source

Trained on augmented 2010 i2b2 challenge data with ‘embeddings_clinical’. https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/

## Benchmarking

```bash
|    | label         |    tp |    fp |    fn |     prec |      rec |       f1 |
|---:|--------------:|------:|------:|------:|---------:|---------:|---------:|
|  0 | I-TREATMENT   |  6625 |  1187 |  1329 | 0.848054 | 0.832914 | 0.840416 |
|  1 | I-PROBLEM     | 15142 |  1976 |  2542 | 0.884566 | 0.856254 | 0.87018  |
|  2 | B-PROBLEM     | 11005 |  1065 |  1587 | 0.911765 | 0.873968 | 0.892466 |
|  3 | I-TEST        |  6748 |   923 |  1264 | 0.879677 | 0.842237 | 0.86055  |
|  4 | B-TEST        |  8196 |   942 |  1029 | 0.896914 | 0.888455 | 0.892665 |
|  5 | B-TREATMENT   |  8271 |  1265 |  1073 | 0.867345 | 0.885167 | 0.876165 |
|  6 | Macro-average | 55987 |  7358 |  8824 | 0.881387 | 0.863166 | 0.872181 |
|  7 | Micro-average | 55987 |  7358 |  8824 | 0.883842 | 0.86385  | 0.873732 |
```