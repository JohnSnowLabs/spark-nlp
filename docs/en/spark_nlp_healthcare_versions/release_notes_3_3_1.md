---
layout: docs
header: true
seotitle: Spark NLP for Healthcare | John Snow Labs
title: Spark NLP for Healthcare Release Notes 3.3.1
permalink: /docs/en/spark_nlp_healthcare_versions/release_notes_3_3_1
key: docs-licensed-release-notes
modify_date: 2021-07-14
show_nav: true
sidebar:
    nav: sparknlp-healthcare
---

## 3.3.1
We are glad to announce that Spark NLP Healthcare 3.3.1 has been released!.

#### Highlights
+ New ChunkKeyPhraseExtraction Annotator
+ New BERT-Based NER Models
+ New UMLS Sentence Entity Resolver Models
+ Updated RxNorm Entity Resolver Model (Dropping Invalid Codes)
+ New showVersion() Method in Compatibility Class
+ New Docker Images for Spark NLP for Healthcare and Spark OCR
+ New and Updated Deidentification() Parameters
+ New Python API Documentation
+ Updated Spark NLP For Healthcare Notebooks and New Notebooks

#### New ChunkKeyPhraseExtraction Annotator

We are releasing `ChunkKeyPhraseExtraction` annotator that leverages Sentence BERT embeddings to select keywords and key phrases that are most similar to a document. This annotator can be fed by either the output of NER model, NGramGenerator or YAKE, and could be used to generate similarity scores for each NER chunk that is coming out of any (clinical) NER model. That is, you can now sort your clinical entities by the importance of them with respect to document or sentence that they live in. Additionally, you can also use this new annotator to grab new clinical chunks that are missed by a pretrained NER model as well as summarizing the whole document into a few important sentences or phrases.

You can find more examples in [ChunkKeyPhraseExtraction notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/9.Chunk_Key_Phrase_Extraction.ipynb)

*Example* :

```bash
...
ngram_ner_key_phrase_extractor = ChunkKeyPhraseExtraction.pretrained("sbert_jsl_medium_uncased ", "en", "clinical/models")\
    .setTopN(5) \
    .setDivergence(0.4)\
    .setInputCols(["sentences", "merged_chunks"])\
    .setOutputCol("key_phrases")
...

text = "A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus ( T2DM ), one prior episode of HTG-induced pancreatitis three years prior to presentation , associated with an acute hepatitis , and obesity with a body mass index ( BMI ) of 33.5 kg/m2 , presented with a one-week history of polyuria , polydipsia , poor appetite , and vomiting . Two weeks prior to presentation, she was treated with a five-day course of amoxicillin for a respiratory tract infection. She was on metformin , glipizide , and dapagliflozin for T2DM and atorvastatin and gemfibrozil for HTG. She had been on dapagliflozin for six months at the time of presentation . Physical examination on presentation was significant for dry oral mucosa ; significantly, her abdominal examination was benign with no tenderness , guarding , or rigidity . Pertinent laboratory findings on admission were: serum glucose 111 mg/dl , bicarbonate 18 mmol/l , anion gap 20 , creatinine 0.4 mg/dL , triglycerides 508 mg/dL , total cholesterol 122 mg/dL , glycated hemoglobin ( HbA1c ) 10% , and venous pH 7.27. Serum lipase was normal at 43 U/L . Serum acetone levels could not be assessed as blood samples kept hemolyzing due to significant lipemia ."


textDF = spark.createDataFrame([[text]]).toDF("text")
ngram_ner_results =  ngram_ner_pipeline.transform(textDF)
```

*Results* :

```bash
+--------------------------+------+-------------------+-------------------+--------+
|key_phrase                |source|DocumentSimilarity |MMRScore           |sentence|
+--------------------------+------+-------------------+-------------------+--------+
|type two diabetes mellitus|NER   |0.7639750686118073 |0.4583850593816694 |0       |
|HTG-induced pancreatitis  |ngrams|0.66933222897749   |0.10416352343367463|0       |
|vomiting                  |ngrams|0.5824238088130589 |0.14864183399720493|0       |
|history polyuria          |ngrams|0.46337313737310987|0.0959500325843913 |0       |
|28-year-old female        |ngrams|0.31692529374916967|0.10043002919664669|0       |
+--------------------------+------+-------------------+-------------------+--------+
```

#### New BERT-Based NER Models

We have two new BERT-Based token classifier NER models.

+ `bert_token_classifier_ner_chemicals` : This model is BERT-based version of `ner_chemicals` model and can detect chemical compounds (`CHEM`) in the medical texts.

*Metrics* :

```bash
              precision    recall  f1-score   support
      B-CHEM       0.94      0.92      0.93     30731
      I-CHEM       0.95      0.93      0.94     31270
    accuracy                           0.99     62001
   macro avg       0.96      0.95      0.96     62001
weighted avg       0.99      0.93      0.96     62001
```


*Example* :

```python
...
tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_ner_chemicals", "en", "clinical/models")\
    .setInputCols("token", "document")\
    .setOutputCol("ner")\
    .setCaseSensitive(True)
...

test_sentence = """The results have shown that the product p - choloroaniline is not a significant factor in chlorhexidine - digluconate associated erosive cystitis. A high percentage of kanamycin - colistin and povidone - iodine irrigations were associated with erosive cystitis."""
result = p_model.transform(spark.createDataFrame([[test_sentence]]).toDF("text"))
```

*Results* :

```bash
+---------------------------+---------+
|chunk                      |ner_label|
+---------------------------+---------+
|p - choloroaniline         |CHEM     |
|chlorhexidine - digluconate|CHEM     |
|kanamycin                  |CHEM     |
|colistin                   |CHEM     |
|povidone - iodine          |CHEM     |
+---------------------------+---------+
```

+ `bert_token_classifier_ner_chemprot` : This model is BERT-based version of `ner_chemprot_clinical` model and can detect chemical compounds and genes (`CHEMICAL`, `GENE-Y`, `GENE-N`) in the medical texts.

*Metrics* :

```bash
              precision    recall  f1-score   support
  B-CHEMICAL       0.80      0.79      0.80      8649
    B-GENE-N       0.53      0.56      0.54      2752
    B-GENE-Y       0.71      0.73      0.72      5490
  I-CHEMICAL       0.82      0.79      0.81      1313
    I-GENE-N       0.62      0.62      0.62      1993
    I-GENE-Y       0.75      0.72      0.74      2420
    accuracy                           0.96     22617
   macro avg       0.75      0.74      0.75     22617
weighted avg       0.83      0.73      0.78     22617
```

*Example* :

```bash
...
tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_ner_chemprot", "en", "clinical/models")\
    .setInputCols("token", "document")\
    .setOutputCol("ner")\
    .setCaseSensitive(True)
...

test_sentence = "Keratinocyte growth factor and acidic fibroblast growth factor are mitogens for primary cultures of mammary epithelium."
result = p_model.transform(spark.createDataFrame([[test_sentence]]).toDF("text"))
```

*Results* :

```bash
+-------------------------------+---------+
|chunk                          |ner_label|
+-------------------------------+---------+
|Keratinocyte growth factor     |GENE-Y   |
|acidic fibroblast growth factor|GENE-Y   |
+-------------------------------+---------+
```


#### New UMLS Sentence Entity Resolver Models

We are releasing two new UMLS Sentence Entity Resolver models trained on 2021AB UMLS dataset and map clinical entities to UMLS CUI codes.

+ `sbiobertresolve_umls_disease_syndrome` : This model is trained on the `Disease` or `Syndrome` category using `sbiobert_base_cased_mli` embeddings.

*Example* :

```bash
...
resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_umls_disease_syndrome","en", "clinical/models") \
     .setInputCols(["ner_chunk", "sbert_embeddings"]) \
     .setOutputCol("resolution")\
     .setDistanceFunction("EUCLIDEAN")
...

data = spark.createDataFrame([["""A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus (T2DM), one prior episode of HTG-induced pancreatitis three years prior to presentation, associated with an acute hepatitis, and obesity with a body mass index (BMI) of 33.5 kg/m2, presented with a one-week history of polyuria, polydipsia, poor appetite, and vomiting."""]]).toDF("text")
results = model.fit(data).transform(data)

```

*Results* :

```bash
|    | chunk                                 | code     | code_description                      | all_k_codes                                                  | all_k_codes_desc                                                                                                                                                                                         |
|---:|:--------------------------------------|:---------|:--------------------------------------|:-------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|  0 | gestational diabetes mellitus         | C0085207 | gestational diabetes mellitus         | ['C0085207', 'C0032969', 'C2063017', 'C1283034', 'C0271663'] | ['gestational diabetes mellitus', 'pregnancy diabetes mellitus', 'pregnancy complicated by diabetes mellitus', 'maternal diabetes mellitus', 'gestational diabetes mellitus, a2']                        |
|  1 | subsequent type two diabetes mellitus | C0348921 | pre-existing type 2 diabetes mellitus | ['C0348921', 'C1719939', 'C0011860', 'C0877302', 'C0271640'] | ['pre-existing type 2 diabetes mellitus', 'disorder associated with type 2 diabetes mellitus', 'diabetes mellitus, type 2', 'insulin-requiring type 2 diabetes mellitus', 'secondary diabetes mellitus'] |
|  2 | HTG-induced pancreatitis              | C0376670 | alcohol-induced pancreatitis          | ['C0376670', 'C1868971', 'C4302243', 'C0267940', 'C2350449'] | ['alcohol-induced pancreatitis', 'toxic pancreatitis', 'igg4-related pancreatitis', 'hemorrhage pancreatitis', 'graft pancreatitis']                                                                     |
|  3 | an acute hepatitis                    | C0019159 | acute hepatitis                       | ['C0019159', 'C0276434', 'C0267797', 'C1386146', 'C2063407'] | ['acute hepatitis a', 'acute hepatitis a', 'acute hepatitis', 'acute infectious hepatitis', 'acute hepatitis e']                                                                                         |
|  4 | obesity                               | C0028754 | obesity                               | ['C0028754', 'C0342940', 'C0342942', 'C0857116', 'C1561826'] | ['obesity', 'abdominal obesity', 'generalized obesity', 'obesity gross', 'overweight and obesity']                                                                                                       |
|  5 | polyuria                              | C0018965 | hematuria                             | ['C0018965', 'C0151582', 'C3888890', 'C0268556', 'C2936921'] | ['hematuria', 'uricosuria', 'polyuria-polydipsia syndrome', 'saccharopinuria', 'saccharopinuria']                                                                                                        |
|  6 | polydipsia                            | C0268813 | primary polydipsia                    | ['C0268813', 'C0030508', 'C3888890', 'C0393777', 'C0206085'] | ['primary polydipsia', 'parasomnia', 'polyuria-polydipsia syndrome', 'hypnogenic paroxysmal dystonias', 'periodic hypersomnias']                                                                         |
|  7 | poor appetite                         | C0003123 | lack of appetite                      | ['C0003123', 'C0011168', 'C0162429', 'C1282895', 'C0039338'] | ['lack of appetite', 'poor swallowing', 'poor nutrition', 'neurologic unpleasant taste', 'taste dis']                                                                                                    |
|  8 | vomiting                              | C0152164 | periodic vomiting                     | ['C0152164', 'C0267172', 'C0152517', 'C0011119', 'C0152227'] | ['periodic vomiting', 'habit vomiting', 'viral vomiting', 'choking', 'tearing']                                                                                                                          |
```

+ `sbiobertresolve_umls_clinical_drugs` : This model is trained on the `Clinical Drug` category using `sbiobert_base_cased_mli` embeddings.

*Example* :

```bash
...
resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_umls_clinical_drugs","en", "clinical/models") \
     .setInputCols(["ner_chunk", "sbert_embeddings"]) \
     .setOutputCol("resolution")\
     .setDistanceFunction("EUCLIDEAN")
...

data = spark.createDataFrame([["""She was immediately given hydrogen peroxide 30 mg to treat the infection on her leg, and has been advised Neosporin Cream for 5 days. She has a history of taking magnesium hydroxide 100mg/1ml and metformin 1000 mg."""]]).toDF("text")
results = model.fit(data).transform(data)

```

*Results* :

```bash
|    | chunk                         | code     | code_description           | all_k_codes                                                  | all_k_codes_desc                                                                                                                                                                        |
|---:|:------------------------------|:---------|:---------------------------|:-------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|  0 | hydrogen peroxide 30 mg       | C1126248 | hydrogen peroxide 30 mg/ml | ['C1126248', 'C0304655', 'C1605252', 'C0304656', 'C1154260'] | ['hydrogen peroxide 30 mg/ml', 'hydrogen peroxide solution 30%', 'hydrogen peroxide 30 mg/ml [proxacol]', 'hydrogen peroxide 30 mg/ml cutaneous solution', 'benzoyl peroxide 30 mg/ml'] |
|  1 | Neosporin Cream               | C0132149 | neosporin cream            | ['C0132149', 'C0358174', 'C0357999', 'C0307085', 'C0698810'] | ['neosporin cream', 'nystan cream', 'nystadermal cream', 'nupercainal cream', 'nystaform cream']                                                                                        |
|  2 | magnesium hydroxide 100mg/1ml | C1134402 | magnesium hydroxide 100 mg | ['C1134402', 'C1126785', 'C4317023', 'C4051486', 'C4047137'] | ['magnesium hydroxide 100 mg', 'magnesium hydroxide 100 mg/ml', 'magnesium sulphate 100mg/ml injection', 'magnesium sulfate 100 mg', 'magnesium sulfate 100 mg/ml']                     |
|  3 | metformin 1000 mg             | C0987664 | metformin 1000 mg          | ['C0987664', 'C2719784', 'C0978482', 'C2719786', 'C4282269'] | ['metformin 1000 mg', 'metformin hydrochloride 1000 mg', 'metformin hcl 1000mg tab', 'metformin hydrochloride 1000 mg [fortamet]', 'metformin hcl 1000mg sa tab']                       |

```

#### Updated RxNorm Entity Resolver Model (Dropping Invalid Codes)

`sbiobertresolve_rxnorm` model was updated by dropping invalid codes using 02 August 2021 RxNorm dataset.

#### New showVersion() Method in Compatibility Class

We added the `.showVersion()` method in our Compatibility class that shows the name of the models and the version in a pretty way.

```python
compatibility = Compatibility()
compatibility.showVersion('sentence_detector_dl_healthcare')
```
After the execution you will see the following table,

```bash
+---------------------------------+------+---------+
| Pipeline/Model                  | lang | version |
+---------------------------------+------+---------+
| sentence_detector_dl_healthcare |  en  | 2.6.0   |
| sentence_detector_dl_healthcare |  en  | 2.7.0   |
| sentence_detector_dl_healthcare |  en  | 3.2.0   |
+---------------------------------+------+---------+
```

#### New Docker Images for Spark NLP for Healthcare and Spark OCR

We are releasing new Docker Images for Spark NLP for Healthcare and Spark OCR containing a jupyter environment. Users having a valid license can run the image on their local system, and connect to pre-configured jupyter instance without installing the library on their local system.

**Spark NLP for Healthcare Docker Image**

For running Spark NLP for Healthcare inside a container:

- Instructions: [Spark NLP for Healthcare Docker Image](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/jupyter/docker_image_nlp_hc)

- Video Instructions: [Youtube Video](https://www.youtube.com/watch?v=tgN0GZGMVJk)

**Spark NLP for Healthcare & OCR Docker Image**

For users who want to run Spark OCR and then feed the output of OCR pipeline to healthcare modules to process further:

- Instructions: [Spark NLP for Healthcare & OCR Docker Image](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/jupyter/docker_image_ocr)

#### New and Updated Deidentification() Parameters

*New Parameter* :
+ `setBlackList()` : List of entities **ignored** for masking or obfuscation.The default values are: `SSN`, `PASSPORT`, `DLN`, `NPI`, `C_CARD`, `IBAN`, `DEA`.

*Updated Parameter* :

+ `.setObfuscateRefSource()` : It was set `faker` as default.

#### New Python API Documentation

We have new Spark NLP for Healthcare [Python API Documentation](https://nlp.johnsnowlabs.com/licensed/api/python/) . This page contains information how to use the library with Python examples.

#### Updated Spark NLP For Healthcare Notebooks and New Notebooks

- New [BertForTokenClassification NER Model Training with Transformers Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.6.BertForTokenClassification_NER_SparkNLP_with_Transformers.ipynb) for showing how to train a BertForTokenClassification NER model with transformers and then import into Spark NLP.

- New [ChunkKeyPhraseExtraction notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/9.Chunk_Key_Phrase_Extraction.ipynb) for showing how to get chunk key phrases using `ChunkKeyPhraseExtraction`.

- Updated all [Spark NLP For Healthcare Notebooks](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings/Healthcare) with v3.3.0 by adding the new features.



**To see more, please check : [Spark NLP Healthcare Workshop Repo](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings/Healthcare)**

<div class="prev_ver h3-box" markdown="1">

## Versions

</div>

<ul class="pagination">
    <li>
        <a href="release_notes_3_3_0">Version 3.3.0</a>
    </li>
    <li>
        <strong>Version 3.3.1</strong>
    </li>
    <li>
        <a href="release_notes_3_3_2">Version 3.3.2</a>
    </li>
</ul>

<ul class="pagination owl-carousel pagination_big">
    <li><a href="release_notes_4_2_1">4.2.1</a></li>
    <li><a href="release_notes_4_2_0">4.2.0</a></li>
    <li><a href="release_notes_4_1_0">4.1.0</a></li>
    <li><a href="release_notes_4_0_2">4.0.2</a></li>
    <li><a href="release_notes_4_0_0">4.0.0</a></li>
    <li><a href="release_notes_3_5_3">3.5.3</a></li>
    <li><a href="release_notes_3_5_2">3.5.2</a></li>
    <li><a href="release_notes_3_5_1">3.5.1</a></li>
    <li><a href="release_notes_3_5_0">3.5.0</a></li>
    <li><a href="release_notes_3_4_2">3.4.2</a></li>
    <li><a href="release_notes_3_4_1">3.4.1</a></li>
    <li><a href="release_notes_3_4_0">3.4.0</a></li>
    <li><a href="release_notes_3_3_4">3.3.4</a></li>
    <li><a href="release_notes_3_3_2">3.3.2</a></li>
    <li class="active"><a href="release_notes_3_3_1">3.3.1</a></li>
    <li><a href="release_notes_3_3_0">3.3.0</a></li>
    <li><a href="release_notes_3_2_3">3.2.3</a></li>
    <li><a href="release_notes_3_2_2">3.2.2</a></li>
    <li><a href="release_notes_3_2_1">3.2.1</a></li>
    <li><a href="release_notes_3_2_0">3.2.0</a></li>
    <li><a href="release_notes_3_1_3">3.1.3</a></li>
    <li><a href="release_notes_3_1_2">3.1.2</a></li>
    <li><a href="release_notes_3_1_1">3.1.1</a></li>
    <li><a href="release_notes_3_1_0">3.1.0</a></li>
    <li><a href="release_notes_3_0_3">3.0.3</a></li>
    <li><a href="release_notes_3_0_2">3.0.2</a></li>
    <li><a href="release_notes_3_0_1">3.0.1</a></li>
    <li><a href="release_notes_3_0_0">3.0.0</a></li>
    <li><a href="release_notes_2_7_6">2.7.6</a></li>
    <li><a href="release_notes_2_7_5">2.7.5</a></li>
    <li><a href="release_notes_2_7_4">2.7.4</a></li>
    <li><a href="release_notes_2_7_3">2.7.3</a></li>
    <li><a href="release_notes_2_7_2">2.7.2</a></li>
    <li><a href="release_notes_2_7_1">2.7.1</a></li>
    <li><a href="release_notes_2_7_0">2.7.0</a></li>
    <li><a href="release_notes_2_6_2">2.6.2</a></li>
    <li><a href="release_notes_2_6_0">2.6.0</a></li>
    <li><a href="release_notes_2_5_5">2.5.5</a></li>
    <li><a href="release_notes_2_5_3">2.5.3</a></li>
    <li><a href="release_notes_2_5_2">2.5.2</a></li>
    <li><a href="release_notes_2_5_0">2.5.0</a></li>
    <li><a href="release_notes_2_4_6">2.4.6</a></li>
    <li><a href="release_notes_2_4_5">2.4.5</a></li>
    <li><a href="release_notes_2_4_2">2.4.2</a></li>
    <li><a href="release_notes_2_4_1">2.4.1</a></li>
    <li><a href="release_notes_2_4_0">2.4.0</a></li>
</ul>