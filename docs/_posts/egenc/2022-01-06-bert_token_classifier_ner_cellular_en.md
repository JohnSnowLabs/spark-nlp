---
layout: model
title: Detect Cellular/Molecular Biology Entities (BertForTokenClassification)
author: John Snow Labs
name: bert_token_classifier_ner_cellular
date: 2022-01-06
tags: [bertfortokenclassification, ner, cellular, en, licensed]
task: Named Entity Recognition
language: en
edition: Spark NLP for Healthcare 3.3.4
spark_version: 2.4
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model detects molecular biology-related terms in medical texts. This model is trained with the `BertForTokenClassification` method from the `transformers` library and imported into Spark NLP.

## Predicted Entities

`DNA`, `Cell_type`, `Cell_line`, `RNA`, `Protein`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_cellular_en_3.3.4_2.4_1641455594142.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
...
tokenClassifier = MedicalBertForTokenClassifier.pretrained("bert_token_classifier_ner_cellular", "en", "clinical/models")\
  .setInputCols("token", "document")\
  .setOutputCol("ner")\
  .setCaseSensitive(True)

ner_converter = NerConverter()\
  .setInputCols(["document","token","ner"])\
  .setOutputCol("ner_chunk")

pipeline = Pipeline(stages=[documentAssembler, sentence_detector, tokenizer, tokenClassifier, ner_converter])

p_model = pipeline.fit(spark.createDataFrame(pd.DataFrame({'text': ['']})))

test_sentence = """Detection of various other intracellular signaling proteins is also described. Genetic characterization of transactivation of the human T-cell leukemia virus type 1 promoter: Binding of Tax to Tax-responsive element 1 is mediated by the cyclic AMP-responsive members of the CREB/ATF family of transcription factors. To achieve a better understanding of the mechanism of transactivation by Tax of human T-cell leukemia virus type 1 Tax-responsive element 1 (TRE-1), we developed a genetic approach with Saccharomyces cerevisiae. We constructed a yeast reporter strain containing the lacZ gene under the control of the CYC1 promoter associated with three copies of TRE-1. Expression of either the cyclic AMP response element-binding protein (CREB) or CREB fused to the GAL4 activation domain (GAD) in this strain did not modify the expression of the reporter gene. Tax alone was also inactive."""

result = p_model.transform(spark.createDataFrame(pd.DataFrame({'text': [test_sentence]})))
```
```scala
val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val tokenizer = new Tokenizer()
    .setInputCols(Array("document"))
    .setOutputCol("token")

val tokenClassifier = MedicalBertForTokenClassifier.pretrained("bert_token_classifier_ner_cellular", "en", "clinical/models")
    .setInputCols(Array("document","token"))
    .setOutputCol("ner")
    .setCaseSensitive(True)

val ner_converter = new NerConverter()
    .setInputCols(Array("document","token","ner"))
    .setOutputCol("ner_chunk")

val pipeline =  new Pipeline().setStages(Array(documentAssembler, tokenizer, tokenClassifier, ner_converter))

val data = Seq("Detection of various other intracellular signaling proteins is also described. Genetic characterization of transactivation of the human T-cell leukemia virus type 1 promoter: Binding of Tax to Tax-responsive element 1 is mediated by the cyclic AMP-responsive members of the CREB/ATF family of transcription factors. To achieve a better understanding of the mechanism of transactivation by Tax of human T-cell leukemia virus type 1 Tax-responsive element 1 (TRE-1), we developed a genetic approach with Saccharomyces cerevisiae. We constructed a yeast reporter strain containing the lacZ gene under the control of the CYC1 promoter associated with three copies of TRE-1. Expression of either the cyclic AMP response element-binding protein (CREB) or CREB fused to the GAL4 activation domain (GAD) in this strain did not modify the expression of the reporter gene. Tax alone was also inactive.").toDF("text")

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
|Model Name:|bert_token_classifier_ner_cellular|
|Compatibility:|Spark NLP for Healthcare 3.3.4+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|404.3 MB|
|Case sensitive:|true|
|Max sentense length:|512|

## Data Source

Trained on Cancer Genetics (CG) task of the BioNLP Shared Task 2013. https://aclanthology.org/W13-2008/

## Benchmarking

```bash
                                   precision    recall  f1-score   support

                     B-Amino_acid       0.77      0.16      0.27        62
              B-Anatomical_system       0.75      0.18      0.29        17
                         B-Cancer       0.88      0.82      0.85       924
                           B-Cell       0.84      0.86      0.85      1013
             B-Cellular_component       0.87      0.84      0.86       180
B-Developing_anatomical_structure       0.65      0.65      0.65        17
           B-Gene_or_gene_product       0.62      0.79      0.69      2520
   B-Immaterial_anatomical_entity       0.68      0.74      0.71        31
         B-Multi-tissue_structure       0.84      0.76      0.80       303
                          B-Organ       0.78      0.74      0.76       156
                       B-Organism       0.93      0.86      0.89       518
           B-Organism_subdivision       0.74      0.51      0.61        39
             B-Organism_substance       0.93      0.66      0.77       102
         B-Pathological_formation       0.85      0.60      0.71        88
                B-Simple_chemical       0.61      0.75      0.68       727
                         B-Tissue       0.74      0.83      0.78       184
                     I-Amino_acid       0.60      1.00      0.75         3
              I-Anatomical_system       1.00      0.11      0.20         9
                         I-Cancer       0.91      0.69      0.78       604
                           I-Cell       0.98      0.74      0.84      1091
             I-Cellular_component       0.88      0.62      0.73        69
I-Developing_anatomical_structure       0.00      0.00      0.00         4
           I-Gene_or_gene_product       0.96      0.27      0.42      2354
   I-Immaterial_anatomical_entity       0.38      0.30      0.33        10
         I-Multi-tissue_structure       0.89      0.86      0.87       162
                          I-Organ       0.67      0.59      0.62        17
                       I-Organism       0.84      0.45      0.59       120
           I-Organism_subdivision       0.00      0.00      0.00         9
             I-Organism_substance       0.80      0.50      0.62        24
         I-Pathological_formation       0.81      0.56      0.67        39
                I-Simple_chemical       0.92      0.15      0.26       622
                         I-Tissue       0.83      0.86      0.84       111
                                O       0.00      0.00      0.00         0

                         accuracy                           0.64     12129
                        macro avg       0.73      0.56      0.60     12129
                     weighted avg       0.83      0.64      0.68     12129
```
