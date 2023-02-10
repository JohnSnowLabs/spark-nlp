---
layout: model
title: Detect Cellular/Molecular Biology Entities (BertForTokenClassification)
author: John Snow Labs
name: bert_token_classifier_ner_cellular
date: 2021-11-03
tags: [bertfortokenclassification, ner, cellular, en, clinical, licensed]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 3.3.0
spark_version: 2.4
supported: true
annotator: MedicalBertForTokenClassifier
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
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_BERT_TOKEN_CLASSIFIER.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_cellular_en_3.3.0_2.4_1635938889847.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_cellular_en_3.3.0_2.4_1635938889847.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}


## How to use






<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = DocumentAssembler()\
        .setInputCol("text")\
        .setOutputCol("document")
         
sentence_detector = SentenceDetector()\
        .setInputCols(["document"])\
        .setOutputCol("sentence")

tokenizer = Tokenizer()\
        .setInputCols(["sentence"])\
        .setOutputCol("token")

tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_ner_cellular", "en", "clinical/models")\
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
         
val sentence_detector = new SentenceDetector()
        .setInputCols("document")
        .setOutputCol("sentence")

val tokenizer = new Tokenizer()
        .setInputCols("sentence")
        .setOutputCol("token")

val tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_ner_cellular", "en", "clinical/models")
        .setInputCols(Array("token", "document"))
        .setOutputCol("ner")
        .setCaseSensitive(True)

val ner_converter = new NerConverter()
        .setInputCols(Array("document","token","ner"))
        .setOutputCol("ner_chunk")

val pipeline =  new Pipeline().setStages(Array(documentAssembler, tokenizer, tokenClassifier, ner_converter))

val data = Seq("""Detection of various other intracellular signaling proteins is also described. Genetic characterization of transactivation of the human T-cell leukemia virus type 1 promoter: Binding of Tax to Tax-responsive element 1 is mediated by the cyclic AMP-responsive members of the CREB/ATF family of transcription factors. To achieve a better understanding of the mechanism of transactivation by Tax of human T-cell leukemia virus type 1 Tax-responsive element 1 (TRE-1), we developed a genetic approach with Saccharomyces cerevisiae. We constructed a yeast reporter strain containing the lacZ gene under the control of the CYC1 promoter associated with three copies of TRE-1. Expression of either the cyclic AMP response element-binding protein (CREB) or CREB fused to the GAL4 activation domain (GAD) in this strain did not modify the expression of the reporter gene. Tax alone was also inactive.""").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.classify.token_bert.cellular").predict("""Detection of various other intracellular signaling proteins is also described. Genetic characterization of transactivation of the human T-cell leukemia virus type 1 promoter: Binding of Tax to Tax-responsive element 1 is mediated by the cyclic AMP-responsive members of the CREB/ATF family of transcription factors. To achieve a better understanding of the mechanism of transactivation by Tax of human T-cell leukemia virus type 1 Tax-responsive element 1 (TRE-1), we developed a genetic approach with Saccharomyces cerevisiae. We constructed a yeast reporter strain containing the lacZ gene under the control of the CYC1 promoter associated with three copies of TRE-1. Expression of either the cyclic AMP response element-binding protein (CREB) or CREB fused to the GAL4 activation domain (GAD) in this strain did not modify the expression of the reporter gene. Tax alone was also inactive.""")
```

</div>


## Results


```bash
+-------------------------------------------+---------+
|chunk                                      |ner_label|
+-------------------------------------------+---------+
|intracellular signaling proteins           |protein  |
|human T-cell leukemia virus type 1 promoter|DNA      |
|Tax                                        |protein  |
|Tax-responsive element 1                   |DNA      |
|cyclic AMP-responsive members              |protein  |
|CREB/ATF family                            |protein  |
|transcription factors                      |protein  |
|Tax                                        |protein  |
|human T-cell leukemia virus type 1         |DNA      |
|Tax-responsive element 1                   |DNA      |
|TRE-1                                      |DNA      |
|lacZ gene                                  |DNA      |
|CYC1 promoter                              |DNA      |
|TRE-1                                      |DNA      |
|cyclic AMP response element-binding protein|protein  |
|CREB                                       |protein  |
|CREB                                       |protein  |
|GAL4 activation domain                     |protein  |
|GAD                                        |protein  |
|reporter gene                              |DNA      |
|Tax                                        |protein  |
+-------------------------------------------+---------+
```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_ner_cellular|
|Compatibility:|Healthcare NLP 3.3.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[ner]|
|Language:|en|
|Case sensitive:|true|
|Max sentense length:|512|


## Data Source


Trained on the JNLPBA corpus containing more than 2.404 publication abstracts. http://www.geniaproject.org/


## Benchmarking


```bash
label           precision    recall   f1-score   support
B-DNA             0.87        0.77      0.82      1056
B-RNA             0.85        0.79      0.82       118
B-cell_line       0.66        0.70      0.68       500
B-cell_type       0.87        0.75      0.81      1921
B-protein         0.90        0.85      0.88      5067
I-DNA             0.93        0.86      0.90      1789
I-RNA             0.92        0.84      0.88       187
I-cell_line       0.67        0.76      0.71       989
I-cell_type       0.92        0.76      0.84      2991
I-protein         0.94        0.80      0.87      4774
accuracy           -           -        0.80     19392
macro-avg         0.76        0.81      0.78     19392
weighted-avg      0.89        0.80      0.85     19392
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTU0NDk4OTcxOF19
-->