---
layout: docs
header: true
seotitle: Spark NLP for Healthcare | John Snow Labs
title: Spark NLP for Healthcare Release Notes 3.1.2
permalink: /docs/en/spark_nlp_healthcare_versions/release_notes_3_1_2
key: docs-licensed-release-notes
modify_date: 2021-07-14
show_nav: true
sidebar:
    nav: sparknlp-healthcare
---

## 3.1.2
We are glad to announce that Spark NLP for Healthcare 3.1.2 has been released!.
This release comes with new features, new models, bug fixes, and examples.

#### Highlights
+ Support for Fine-tuning of Ner models.
+ More builtin(pre-defined) graphs for MedicalNerApproach.
+ Date Normalizer.
+ New Relation Extraction Models for ADE.
+ Bug Fixes.
+ Support for user-defined Custom Transformer.
+ Java Workshop Examples.
+ Deprecated Compatibility class in Python.


##### Support for Fine Tuning of Ner models

Users can now resume training/fine-tune existing(already trained) Spark NLP MedicalNer models on new data. Users can simply provide the path to any existing MedicalNer model and train it further on the new dataset:

```
ner_tagger = MedicalNerApproach().setPretrainedModelPath("/path/to/trained/medicalnermodel")
```

If the new dataset contains new tags/labels/entities, users can choose to override existing tags with the new ones. The default behaviour is to reset the list of existing tags and generate a new list from the new dataset. It is also possible to preserve the existing tags by setting the 'overrideExistingTags' parameter:

```
ner_tagger = MedicalNerApproach()\
  .setPretrainedModelPath("/path/to/trained/medicalnermodel")\
  .setOverrideExistingTags(False)
```

Setting overrideExistingTags to false is intended to be used when resuming trainig on the same, or very similar dataset (i.e. with the same tags or with just a few different ones).

If tags overriding is disabled, and new tags are found in the training set, then the approach will try to allocate them to unused output nodes, if any. It is also possible to override specific tags of the old model by mapping them to new tags:


```bash
ner_tagger = MedicalNerApproach()\
  .setPretrainedModelPath("/path/to/trained/medicalnermodel")\
  .setOverrideExistingTags(False)\
  .setTagsMapping("B-PER,B-VIP", "I-PER,I-VIP")
```

In this case, the new tags `B-VIP` and `I-VIP` will replace the already trained tags 'B-PER' and 'I-PER'. Unmapped old tags will remain in use and unmapped new tags will be allocated to new outpout nodes, if any.

Jupyter Notebook: [Finetuning Medical NER Model Notebook] (https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.5.Resume_MedicalNer_Model_Training.ipynb)

##### More builtin graphs for MedicalNerApproach

Seventy new TensorFlow graphs have been added to the library of available graphs which are used to train MedicalNer models. The graph with the optimal set of parameters is automatically chosen by MedicalNerApproach.


##### DateNormalizer

New annotator that normalize dates to the format YYYY/MM/DD.
This annotator identifies dates in chunk annotations, and transform these dates to the format YYYY/MM/DD.
Both the input and output formats for the annotator are `chunk`.

Example:


```bash
	sentences = [
		    ['08/02/2018'],
		    ['11/2018'],
		    ['11/01/2018'],
		    ['12Mar2021'],
		    ['Jan 30, 2018'],
		    ['13.04.1999'],
		    ['3April 2020'],
		    ['next monday'],
		    ['today'],
		    ['next week'],
	]
	df = spark.createDataFrame(sentences).toDF("text")
	document_assembler = DocumentAssembler().setInputCol('text').setOutputCol('document')
	chunksDF = document_assembler.transform(df)
	aa = map_annotations_col(chunksDF.select("document"),
				    lambda x: [Annotation('chunk', a.begin, a.end, a.result, a.metadata, a.embeddings) for a in x], "document",
				    "chunk_date", "chunk")
	dateNormalizer = DateNormalizer().setInputCols('chunk_date').setOutputCol('date').setAnchorDateYear(2021).setAnchorDateMonth(2).setAnchorDateDay(27)
	dateDf = dateNormalizer.transform(aa)
	dateDf.select("date.result","text").show()
```

```bash

+-----------+----------+
|text        |  date    |
+-----------+----------+
|08/02/2018  |2018/08/02|
|11/2018     |2018/11/DD|
|11/01/2018  |2018/11/01|
|12Mar2021   |2021/03/12|
|Jan 30, 2018|2018/01/30|
|13.04.1999  |1999/04/13|
|3April 2020 |2020/04/03|
|next Monday |2021/06/19|
|today       |2021/06/12|
|next week   |2021/06/19|
+-----------+----------+
```

##### New Relation Extraction Models for ADE

We are releasing new Relation Extraction models for ADE (Adverse Drug Event). This model is available in both `RelationExtraction` and Bert based `RelationExtractionDL` versions, and is capabale of linking drugs with ADE mentions.

Example

```bash
    ade_re_model = new RelationExtractionModel().pretrained('ner_ade_clinical', 'en', 'clinical/models')\
                                     .setInputCols(["embeddings", "pos_tags", "ner_chunk", "dependencies"])\
                                     .setOutputCol("relations")\
                                     .setPredictionThreshold(0.5)\
                                     .setRelationPairs(['ade-drug', 'drug-ade'])
    pipeline = Pipeline(stages=[documenter, sentencer, tokenizer, pos_tagger, words_embedder, ner_tagger, ner_converter,
                                                   dependency_parser, re_ner_chunk_filter, re_model])
    text ="""A 30 year old female presented with tense bullae due to excessive use of naproxin, and leg cramps relating to oxaprozin."""

    p_model = pipeline.fit(spark.createDataFrame([[text]]).toDF("text"))

    result = p_model.transform(data)

```

Results
```bash
|    | chunk1        | entity1    |  chunk2       | entity2   |    result  |
|---:|:--------------|:-----------|:--------------|:----------|-----------:|
|  0 | tense bullae  | ADE        | naproxin      | DRUG      |          1 |
|  1 | tense bullae  | ADE        | oxaprozin     | DRUG      |          0 |
|  2 | naproxin      | DRUG       | leg cramps    | ADE       |          0 |
|  3 | leg cramps    | ADE        | oxaprozin     | DRUG      |          1 |

```

Benchmarking
Model: `re_ade_clinical`
```bash

              precision    recall  f1-score   support
           0       0.85      0.89      0.87      1670
           1       0.88      0.84      0.86      1673
   micro avg       0.87      0.87      0.87      3343
   macro avg       0.87      0.87      0.87      3343
weighted avg       0.87      0.87      0.87      3343
```

Model: `redl_ade_biobert`
```bash
Relation           Recall Precision        F1   Support
0                   0.894     0.946     0.919      1011
1                   0.963     0.926     0.944      1389
Avg.                0.928     0.936     0.932
Weighted Avg.       0.934     0.934     0.933
```

##### Bug Fixes
+ RelationExtractionDLModel had an issue(BufferOverflowException) on versions 3.1.0 and 3.1.1, which is fixed with this release.
+ Some pretrained RelationExtractionDLModels got outdated after release 3.0.3, new updated models were created, tested and made available to be used with versions 3.0.3, and later.
+ Some SentenceEntityResolverModels which did not work with Spark 2.4/2.3 were fixed.

##### Support for user-defined Custom Transformer.
Utility classes to define custom transformers in python are included in this release. This allows users to define functions in Python to manipulate Spark-NLP annotations. This new Transformers can be added to pipelines like any of the other models you're already familiar with.
Example how to use the custom transformer.

```python
        def myFunction(annotations):
            # lower case the content of the annotations
            return [a.copy(a.result.lower()) for a in annotations]

        custom_transformer = CustomTransformer(f=myFunction).setInputCol("ner_chunk").setOutputCol("custom")
        outputDf = custom_transformer.transform(outdf).select("custom").toPandas()
```

##### Java Workshop Examples

Add Java examples in the workshop repository.
https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/java/healthcare

##### Deprecated Compatibility class in Python

Due to active release cycle, we are adding & training new pretrained models at each release and it might be tricky to maintain the backward compatibility or keep up with the latest models and versions, especially for the users using our models locally in air-gapped networks.

We are releasing a new utility class to help you check your local & existing models with the latest version of everything we have up to date. You will not need to specify your AWS credentials from now on. This new class is now replacing the previous Compatibility class written in Python and CompatibilityBeta class written in Scala.

```        
from sparknlp_jsl.compatibility import Compatibility

compatibility = Compatibility(spark)

print(compatibility.findVersion('sentence_detector_dl_healthcare'))
```

Output
```
[{'name': 'sentence_detector_dl_healthcare', 'sparkVersion': '2.4', 'version': '2.6.0', 'language': 'en', 'date': '2020-09-13T14:44:42.565', 'readyToUse': 'true'}, {'name': 'sentence_detector_dl_healthcare', 'sparkVersion': '2.4', 'version': '2.7.0', 'language': 'en', 'date': '2021-03-16T08:42:34.391', 'readyToUse': 'true'}]
```

<div class="prev_ver h3-box" markdown="1">

## Versions

</div>

<ul class="pagination">
    <li>
        <a href="release_notes_3_1_1">Version 3.1.1</a>
    </li>
    <li>
        <strong>Version 3.1.2</strong>
    </li>
    <li>
        <a href="release_notes_3_1_3">Version 3.1.3</a>
    </li>
</ul>

<ul class="pagination owl-carousel pagination_big">
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
    <li><a href="release_notes_3_3_1">3.3.1</a></li>
    <li><a href="release_notes_3_3_0">3.3.0</a></li>
    <li><a href="release_notes_3_2_3">3.2.3</a></li>
    <li><a href="release_notes_3_2_2">3.2.2</a></li>
    <li><a href="release_notes_3_2_1">3.2.1</a></li>
    <li><a href="release_notes_3_2_0">3.2.0</a></li>
    <li><a href="release_notes_3_1_3">3.1.3</a></li>
    <li class="active"><a href="release_notes_3_1_2">3.1.2</a></li>
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