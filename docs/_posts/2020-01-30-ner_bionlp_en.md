---
layout: model
title: Detect Cancer Genetics
author: John Snow Labs
name: ner_bionlp_en
date: 2020-01-30
task: Named Entity Recognition
language: en
edition: Healthcare NLP 2.4.0
spark_version: 2.4
tags: [clinical, licensed, ner, en]
supported: true
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description

Pretrained named entity recognition deep learning model for biology and genetics terms. The SparkNLP deep learning model (NerDL) is inspired by a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM-CNN. 

{:.h2_title}
## Predicted Entities 
``Amino_acid``, ``Anatomical_system``, ``Cancer``, ``Cell``, ``Cellular_component``, ``Developing_anatomical_Structure``, ``Gene_or_gene_product``, ``Immaterial_anatomical_entity``, ``Multi-tissue_structure``, ``Organ``, ``Organism``, ``Organism_subdivision``, ``Simple_chemical``, ``Tissue``, ``Organism_substance``, ``Pathological_formation``


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_TUMOR){:.button.button-orange}
[Open in Colab](https://githubtocolab.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}{:target="_blank"}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_bionlp_en_2.4.0_2.4_1580237286004.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_bionlp_en_2.4.0_2.4_1580237286004.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

{:.h2_title}
## How to use
Use as part of an nlp pipeline with the following stages: DocumentAssembler, SentenceDetector, Tokenizer, WordEmbeddingsModel, NerDLModel. Add the NerConverter to the end of the pipeline to convert entity tokens into full entity chunks.

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}


```python
...
word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
  .setInputCols(["sentence", "token"])\
  .setOutputCol("embeddings")
clinical_ner = NerDLModel.pretrained("ner_bionlp", "en", "clinical/models") \
  .setInputCols(["sentence", "token", "embeddings"]) \
  .setOutputCol("ner")
...
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter])
model = nlpPipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

results = model.transform(spark.createDataFrame([["""The human KCNJ9 (Kir 3.3, GIRK3) is a member of the G-protein-activated inwardly rectifying potassium (GIRK) channel family. Here we describe the genomicorganization of the KCNJ9 locus on chromosome 1q21-23 as a candidate gene forType II diabetes mellitus in the Pima Indian population. The gene spansapproximately 7.6 kb and contains one noncoding and two coding exons separated byapproximately 2.2 and approximately 2.6 kb introns, respectively. We identified14 single nucleotide polymorphisms (SNPs), including one that predicts aVal366Ala substitution, and an 8 base-pair (bp) insertion/deletion. Ourexpression studies revealed the presence of the transcript in various humantissues including pancreas, and two major insulin-responsive tissues: fat andskeletal muscle. The characterization of the KCNJ9 gene should facilitate furtherstudies on the function of the KCNJ9 protein and allow evaluation of thepotential role of the locus in Type II diabetes."""]], ["text"]))

```

```scala
...
val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
  .setInputCols(Array("sentence", "token"))
  .setOutputCol("embeddings")
val ner = NerDLModel.pretrained("ner_bionlp", "en", "clinical/models")
  .setInputCols("sentence", "token", "embeddings")
  .setOutputCol("ner")
...
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, ner, ner_converter))

val data = Seq("The human KCNJ9 (Kir 3.3, GIRK3) is a member of the G-protein-activated inwardly rectifying potassium (GIRK) channel family. Here we describe the genomicorganization of the KCNJ9 locus on chromosome 1q21-23 as a candidate gene forType II diabetes mellitus in the Pima Indian population. The gene spansapproximately 7.6 kb and contains one noncoding and two coding exons separated byapproximately 2.2 and approximately 2.6 kb introns, respectively. We identified14 single nucleotide polymorphisms (SNPs), including one that predicts aVal366Ala substitution, and an 8 base-pair (bp) insertion/deletion. Ourexpression studies revealed the presence of the transcript in various humantissues including pancreas, and two major insulin-responsive tissues: fat andskeletal muscle. The characterization of the KCNJ9 gene should facilitate furtherstudies on the function of the KCNJ9 protein and allow evaluation of thepotential role of the locus in Type II diabetes.").toDF("text")
val result = pipeline.fit(data).transform(data)

```

</div>
{:.h2_title}
## Results
The output is a dataframe with a sentence per row and a ``"ner_label"`` column containing all of the entity labels in the sentence, entity character indices, and other metadata. To get only the tokens and entity labels, without the metadata, select ``"token.result"`` and ``"ner.result"`` from your output dataframe or add the ``"Finisher"`` to the end of your pipeline.

```bash
|id |sentence_id|chunk                 |begin|end|ner_label           |
+---+-----------+----------------------+-----+---+--------------------+
|0  |0          |human                 |4    |8  |Organism            |
|0  |0          |Kir 3.3               |17   |23 |Gene_or_gene_product|
|0  |0          |GIRK3                 |26   |30 |Gene_or_gene_product|
|0  |0          |potassium             |92   |100|Simple_chemical     |
|0  |0          |GIRK                  |103  |106|Gene_or_gene_product|
|0  |1          |chromosome 1q21-23    |188  |205|Cellular_component  |
|0  |5          |pancreas              |697  |704|Organ               |
|0  |5          |tissues               |740  |746|Tissue              |
|0  |5          |fat andskeletal muscle|749  |770|Tissue              |
|0  |6          |KCNJ9                 |801  |805|Gene_or_gene_product|
|0  |6          |Type II               |940  |946|Gene_or_gene_product|
|1  |0          |breast cancer         |84   |96 |Cancer              |
|1  |0          |patients              |134  |141|Organism            |
|1  |0          |anthracyclines        |167  |180|Simple_chemical     |
|1  |0          |taxanes               |186  |192|Simple_chemical     |
|1  |1          |vinorelbine           |246  |256|Simple_chemical     |
|1  |1          |patients              |273  |280|Organism            |
|1  |1          |breast                |309  |314|Cancer              |
|1  |1          |vinorelbine inpatients|386  |407|Simple_chemical     |
|1  |1          |anthracyclines        |433  |446|Simple_chemical     |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_bionlp_en_2.4.0_2.4|
|Type:|ner|
|Compatibility:|Spark NLP 2.4.0+|
|Edition:|Official|
|License:|Licensed|
|Input Labels:|[sentence,token, embeddings]|
|Output Labels:|[ner]|
|Language:|[en]|
|Case sensitive:|false|
|Dependency:embeddings_clinical|

{:.h2_title}
## Data Source
Trained on Cancer Genetics (CG) task of the BioNLP Shared Task 2013 with ``embeddings_clinical``.
https://aclanthology.org/W13-2008/

{:.h2_title}
## Benchmarking
```bash
|    | label                             |    tp |    fp |   fn |     prec |      rec |       f1 |
|---:|:----------------------------------|------:|------:|-----:|---------:|---------:|---------:|
|  1 | I-Amino_acid                      |     1 |     0 |    2 | 1        | 0.333333 | 0.5      |
|  2 | I-Simple_chemical                 |   264 |    39 |  358 | 0.871287 | 0.424437 | 0.570811 |
|  3 | B-Immaterial_anatomical_entity    |    19 |    12 |   12 | 0.612903 | 0.612903 | 0.612903 |
|  4 | B-Cellular_component              |   144 |    24 |   36 | 0.857143 | 0.8      | 0.827586 |
|  5 | B-Cancer                          |   808 |   103 |  115 | 0.886937 | 0.875406 | 0.881134 |
|  6 | I-Cell                            |   888 |    91 |  198 | 0.907048 | 0.81768  | 0.860048 |
|  7 | B-Tissue                          |   137 |    44 |   47 | 0.756906 | 0.744565 | 0.750685 |
|  8 | B-Organism_substance              |    67 |     4 |   34 | 0.943662 | 0.663366 | 0.77907  |
|  9 | B-Simple_chemical                 |   598 |   165 |  128 | 0.783748 | 0.823692 | 0.803224 |
| 10 | B-Cell                            |   910 |   125 |   98 | 0.879227 | 0.902778 | 0.890847 |
| 11 | I-Organ                           |     7 |     2 |   10 | 0.777778 | 0.411765 | 0.538462 |
| 12 | I-Tissue                          |    86 |    21 |   25 | 0.803738 | 0.774775 | 0.788991 |
| 13 | I-Pathological_formation          |    20 |     5 |   19 | 0.8      | 0.512821 | 0.625    |
| 14 | I-Organism                        |    58 |    13 |   62 | 0.816901 | 0.483333 | 0.60733  |
| 15 | B-Gene_or_gene_product            |  2354 |   282 |  165 | 0.89302  | 0.934498 | 0.913288 |
| 16 | I-Cancer                          |   488 |    73 |  116 | 0.869875 | 0.807947 | 0.837768 |
| 17 | B-Organ                           |   109 |    36 |   47 | 0.751724 | 0.698718 | 0.724252 |
| 18 | B-Pathological_formation          |    58 |    20 |   30 | 0.74359  | 0.659091 | 0.698795 |
| 19 | I-Cellular_component              |    33 |     5 |   36 | 0.868421 | 0.478261 | 0.616822 |
| 20 | I-Multi-tissue_structure          |   132 |    34 |   29 | 0.795181 | 0.819876 | 0.807339 |
| 21 | B-Organism                        |   437 |    53 |   77 | 0.891837 | 0.850195 | 0.870518 |
| 22 | I-Gene_or_gene_product            |  1268 |   161 | 1086 | 0.887334 | 0.538658 | 0.670367 |
| 23 | B-Multi-tissue_structure          |   228 |    62 |   73 | 0.786207 | 0.757475 | 0.771574 |
| 24 | Macro-average                     | 9159  | 1398  | 2948 | 0.76803  | 0.548396 | 0.639891 |
| 25 | Micro-average                     | 9159  | 1398  | 2948 | 0.867576 | 0.756505 | 0.808242 |
```