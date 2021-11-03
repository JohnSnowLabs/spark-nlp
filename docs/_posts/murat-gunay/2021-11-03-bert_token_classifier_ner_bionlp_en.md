---
layout: model
title: Detect Cancer Genetics (BertForTokenClassification)
author: John Snow Labs
name: bert_token_classifier_ner_bionlp
date: 2021-11-03
tags: [berfortokenclassification, ner, bionlp, cancer, genetic, clinical, en, licensed]
task: Named Entity Recognition
language: en
edition: Spark NLP for Healthcare 3.3.0
spark_version: 2.4
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model extracts biological and genetics terms in cancer-related texts using pre-trained NER model. This model is trained with the `BertForTokenClassification` method from the `transformers` library and imported into Spark NLP.

## Predicted Entities

`Amino_acid`, `Anatomical_system`, `Cancer`, `Cell`, `Cellular_component`, `Developing_anatomical_Structure`, `Gene_or_gene_product`, `Immaterial_anatomical_entity`, `Multi-tissue_structure`, `Organ`, `Organism`, `Organism_subdivision`, `Simple_chemical`, `Tissue`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_BERT_TOKEN_CLASSIFIER.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_bionlp_en_3.3.0_2.4_1635930176192.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
...
tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_ner_bionlp", "en", "clinical/models")
.setInputCols("token", "document")
.setOutputCol("ner")
.setCaseSensitive(True)

ner_converter = NerConverter()
.setInputCols(["document","token","ner"])
.setOutputCol("ner_chunk") pipeline = Pipeline(stages=[documentAssembler, tokenizer, tokenClassifier, ner_converter])

p_model = pipeline.fit(spark.createDataFrame(pd.DataFrame({'text': ['']})))

test_sentence = """Both the erbA IRES and the erbA/myb virus constructs transformed erythroid cells after infection of bone marrow or blastoderm cultures. The erbA/myb IRES virus exhibited a 5-10-fold higher transformed colony forming efficiency than the erbA IRES virus in the blastoderm assay."""

result = p_model.transform(spark.createDataFrame(pd.DataFrame({'text': [test_sentence]})))
```
```scala
...

val tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_ner_bionlp", "en", "clinical/models")
  .setInputCols("token", "document")
  .setOutputCol("ner")
  .setCaseSensitive(True)

val ner_converter = NerConverter()
        .setInputCols(Array("document","token","ner"))
        .setOutputCol("ner_chunk")

val pipeline =  new Pipeline().setStages(Array(documentAssembler, tokenizer, tokenClassifier, ner_converter))

val data = Seq("Both the erbA IRES and the erbA/myb virus constructs transformed erythroid cells after infection of bone marrow or blastoderm cultures. The erbA/myb IRES virus exhibited a 5-10-fold higher transformed colony forming efficiency than the erbA IRES virus in the blastoderm assay.").toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+-------------------+----------------------+
|chunk              |ner_label             |
+-------------------+----------------------+
|erbA IRES          |Organism              |
|erbA/myb virus     |Organism              |
|erythroid cells    |Cell                  |
|bone marrow        |Multi-tissue_structure|
|blastoderm cultures|Cell                  |
|erbA/myb IRES virus|Organism              |
|erbA IRES virus    |Organism              |
|blastoderm         |Cell                  |
+-------------------+----------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_ner_bionlp|
|Compatibility:|Spark NLP for Healthcare 3.3.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[ner]|
|Language:|en|
|Case sensitive:|true|
|Max sentense length:|512|

## Data Source

Trained on Cancer Genetics (CG) task of the BioNLP Shared Task 2013. https://aclanthology.org/W13-2008/

## Benchmarking

```bash
                                   precision    recall  f1-score   support

                     B-Amino_acid       0.82      0.23      0.35        62
              B-Anatomical_system       0.50      0.06      0.11        17
                         B-Cancer       0.88      0.83      0.85       924
                           B-Cell       0.81      0.87      0.84      1013
             B-Cellular_component       0.88      0.83      0.86       180
B-Developing_anatomical_structure       0.60      0.71      0.65        17
           B-Gene_or_gene_product       0.61      0.81      0.70      2520
   B-Immaterial_anatomical_entity       0.63      0.71      0.67        31
         B-Multi-tissue_structure       0.84      0.79      0.81       303
                          B-Organ       0.71      0.72      0.72       156
                       B-Organism       0.92      0.86      0.89       518
           B-Organism_subdivision       0.70      0.18      0.29        39
             B-Organism_substance       0.97      0.68      0.80       102
         B-Pathological_formation       0.80      0.59      0.68        88
                B-Simple_chemical       0.62      0.74      0.68       727
                         B-Tissue       0.71      0.80      0.75       184
                     I-Amino_acid       0.50      0.33      0.40         3
              I-Anatomical_system       1.00      0.11      0.20         9
                         I-Cancer       0.91      0.72      0.80       604
                           I-Cell       0.97      0.74      0.84      1091
             I-Cellular_component       0.87      0.65      0.74        69
I-Developing_anatomical_structure       0.00      0.00      0.00         4
           I-Gene_or_gene_product       0.96      0.28      0.44      2354
   I-Immaterial_anatomical_entity       0.60      0.30      0.40        10
         I-Multi-tissue_structure       0.81      0.88      0.84       162
                          I-Organ       0.60      0.35      0.44        17
                       I-Organism       0.89      0.47      0.62       120
           I-Organism_subdivision       0.00      0.00      0.00         9
             I-Organism_substance       0.91      0.42      0.57        24
         I-Pathological_formation       0.79      0.56      0.66        39
                I-Simple_chemical       0.90      0.16      0.27       622
                         I-Tissue       0.85      0.79      0.82       111
                                O       0.00      0.00      0.00         0

                         accuracy                           0.65     12129
                        macro avg       0.71      0.52      0.57     12129
                     weighted avg       0.82      0.65      0.68     12129
```