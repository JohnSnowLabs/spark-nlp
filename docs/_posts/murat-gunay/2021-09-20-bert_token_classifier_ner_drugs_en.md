---
layout: model
title: Detect Drug Chemicals (BertForTokenClassifier)
author: John Snow Labs
name: bert_token_classifier_ner_drugs
date: 2021-09-20
tags: [drug, ner, en, licensed]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 3.2.0
spark_version: 2.4
supported: true
annotator: MedicalBertForTokenClassifier
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description


Pretrained named entity recognition deep learning model for Drugs. This model is traiend with `BertForTokenClassification` method from `transformers` library and imported into Spark NLP. It detects drug chemicals.


## Predicted Entities


`DrugChem`


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_BERT_TOKEN_CLASSIFIER/){:.button.button-orange}{:target="_blank"}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_BERT_TOKEN_CLASSIFIER.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_drugs_en_3.2.0_2.4_1632141658042.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


## How to use






<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentenceDetector = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence")

tokenizer = Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token")

tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_ner_drugs", "en", "clinical/models")\
    .setInputCols("token", "sentence")\
    .setOutputCol("ner")\
    .setCaseSensitive(True)

ner_converter = NerConverter()\
    .setInputCols(["sentence","token","ner"])\
    .setOutputCol("ner_chunk")

pipeline =  Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, tokenClassifier, ner_converter])

model = pipeline.fit(spark.createDataFrame(pd.DataFrame({'text': ['']})))

test_sentence = """The human KCNJ9 (Kir 3.3, GIRK3) is a member of the G-protein-activated inwardly rectifying potassium (GIRK) channel family. Here we describe the genomicorganization of the KCNJ9 locus on chromosome 1q21-23 as a candidate gene forType II diabetes mellitus in the Pima Indian population. The gene spansapproximately 7.6 kb and contains one noncoding and two coding exons separated byapproximately 2.2 and approximately 2.6 kb introns, respectively. We identified14 single nucleotide polymorphisms (SNPs), including one that predicts aVal366Ala substitution, and an 8 base-pair (bp) insertion/deletion. Ourexpression studies revealed the presence of the transcript in various humantissues including pancreas, and two major insulin-responsive tissues: fat andskeletal muscle. The characterization of the KCNJ9 gene should facilitate furtherstudies on the function of the KCNJ9 protein and allow evaluation of thepotential role of the locus in Type II diabetes.BACKGROUND: At present, it is one of the most important issues for the treatment of breast cancer to develop the standard therapy for patients previously treated with anthracyclines and taxanes. With the objective of determining the usefulnessof vinorelbine monotherapy in patients with advanced or recurrent breast cancerafter standard therapy, we evaluated the efficacy and safety of vinorelbine inpatients previously treated with anthracyclines and taxanes."""

result = model.transform(spark.createDataFrame(pd.DataFrame({'text': [test_sentence]})))
```
```scala
val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val sentenceDetector = new SentenceDetector()
    .setInputCols("document")
    .setOutputCol("sentence")

val tokenizer = new Tokenizer()
    .setInputCols("sentence")
    .setOutputCol("token")

val tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_ner_drugs", "en", "clinical/models")
    .setInputCols(Array("token", "sentence"))
    .setOutputCol("ner")
    .setCaseSensitive(True)

val ner_converter = new NerConverter()
    .setInputCols(Array("sentence","token","ner"))
    .setOutputCol("ner_chunk")

val pipeline =  new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, tokenClassifier, ner_converter))

val data = Seq("""The human KCNJ9 (Kir 3.3, GIRK3) is a member of the G-protein-activated inwardly rectifying potassium (GIRK) channel family. Here we describe the genomicorganization of the KCNJ9 locus on chromosome 1q21-23 as a candidate gene forType II diabetes mellitus in the Pima Indian population. The gene spansapproximately 7.6 kb and contains one noncoding and two coding exons separated byapproximately 2.2 and approximately 2.6 kb introns, respectively. We identified14 single nucleotide polymorphisms (SNPs), including one that predicts aVal366Ala substitution, and an 8 base-pair (bp) insertion/deletion. Ourexpression studies revealed the presence of the transcript in various humantissues including pancreas, and two major insulin-responsive tissues: fat andskeletal muscle. The characterization of the KCNJ9 gene should facilitate furtherstudies on the function of the KCNJ9 protein and allow evaluation of thepotential role of the locus in Type II diabetes.BACKGROUND: At present, it is one of the most important issues for the treatment of breast cancer to develop the standard therapy for patients previously treated with anthracyclines and taxanes. With the objective of determining the usefulnessof vinorelbine monotherapy in patients with advanced or recurrent breast cancerafter standard therapy, we evaluated the efficacy and safety of vinorelbine inpatients previously treated with anthracyclines and taxanes.""").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.classify.token_bert.ner_drugs").predict("""The human KCNJ9 (Kir 3.3, GIRK3) is a member of the G-protein-activated inwardly rectifying potassium (GIRK) channel family. Here we describe the genomicorganization of the KCNJ9 locus on chromosome 1q21-23 as a candidate gene forType II diabetes mellitus in the Pima Indian population. The gene spansapproximately 7.6 kb and contains one noncoding and two coding exons separated byapproximately 2.2 and approximately 2.6 kb introns, respectively. We identified14 single nucleotide polymorphisms (SNPs), including one that predicts aVal366Ala substitution, and an 8 base-pair (bp) insertion/deletion. Ourexpression studies revealed the presence of the transcript in various humantissues including pancreas, and two major insulin-responsive tissues: fat andskeletal muscle. The characterization of the KCNJ9 gene should facilitate furtherstudies on the function of the KCNJ9 protein and allow evaluation of thepotential role of the locus in Type II diabetes.BACKGROUND: At present, it is one of the most important issues for the treatment of breast cancer to develop the standard therapy for patients previously treated with anthracyclines and taxanes. With the objective of determining the usefulnessof vinorelbine monotherapy in patients with advanced or recurrent breast cancerafter standard therapy, we evaluated the efficacy and safety of vinorelbine inpatients previously treated with anthracyclines and taxanes.""")
```

</div>


## Results


```bash
+--------------+---------+
|chunk         |ner_label|
+--------------+---------+
|potassium     |DrugChem |
|nucleotide    |DrugChem |
|anthracyclines|DrugChem |
|taxanes       |DrugChem |
|vinorelbine   |DrugChem |
|vinorelbine   |DrugChem |
|anthracyclines|DrugChem |
|taxanes       |DrugChem |
+--------------+---------+
```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_ner_drugs|
|Compatibility:|Healthcare NLP 3.2.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[ner]|
|Language:|en|
|Case sensitive:|true|
|Max sentense length:|128|


## Data Source


Trained on i2b2_med7 + FDA. https://www.i2b2.org/NLP/Medication


## Benchmarking


```bash
label          precision    recall  f1-score   support
B-DrugChem       0.99        0.99      0.99     97872
I-DrugChem       0.99        0.99      0.99     54909
O                1.00        1.00      1.00   1191109
accuracy          -           -        1.00   1343890
macro-avg        0.99        0.99      0.99   1343890
weighted-avg     1.00        1.00      1.00   1343890
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTIwMzU0ODgxMTNdfQ==
-->