---
layout: model
title: NER on Financial Texts (Generic)
author: John Snow Labs
name: finner_sec_conll
date: 2022-08-03
tags: [en, financial, ner, sec, licensed]
task: Named Entity Recognition
language: en
edition: Spark NLP for Finance 1.0.0
spark_version: 3.2
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model detects Organizations (ORG), People (PER) and Locations (LOC) in financial texts. Was trained using manual annotations, coNll2003 and financial documents obtained from U.S. Security and Exchange Commission (SEC) filings.

Financial documents may be long, going over the limits of most of the standard Deep Learning and Transforming architectures. Please considering aggressive Sentece Splitting mechanisms to split sentences into smaller chunks.

## Predicted Entities

`ORG`, `LOC`, `PER`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finner_sec_conll_en_1.0.0_3.2_1659538248238.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = DocumentAssembler()\
		.setInputCol("text")\
		.setOutputCol("document")

# Consider using SentenceDetector with rules/patterns to get smaller chunks from long sentences
sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "en")\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = Tokenizer()\
		.setInputCols(["sentence"])\
		.setOutputCol("token")
	
embeddings = BertEmbeddings.pretrained("bert_embeddings_legal_bert_base_uncased","en") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("embeddings")
    
jsl_ner = FinancialNerModel.pretrained("finner_sec_conll", "en", "finance/models") \
		.setInputCols(["sentence", "token", "embeddings"]) \
		.setOutputCol("jsl_ner")

jsl_ner_converter = NerConverter() \
		.setInputCols(["sentence", "token", "jsl_ner"]) \
		.setOutputCol("ner_chunk")

jsl_ner_pipeline = Pipeline().setStages([
				documentAssembler,
				sentence_detector,
				tokenizer,
				embeddings,
				jsl_ner,
				jsl_ner_converter])

text = """December 2007 SUBORDINATED LOAN AGREEMENT. THIS LOAN AGREEMENT is made on 7th December, 2007 BETWEEN: (1) SILICIUM DE PROVENCE S.A.S., a private company with limited liability, incorporated under the laws of France, whose registered office is situated at Usine de Saint Auban, France, represented by Mr.Frank Wouters, hereinafter referred to as the "Borrower", and ( 2 ) EVERGREEN SOLAR INC., a company incorporated in Delaware, U.S.A., with registered number 2426798, whose registered office is situated at Bartlett Street, Marlboro, Massachusetts, U.S.A. represented by Richard Chleboski, hereinafter referred to as "Lender"."""

df = spark.createDataFrame([[text]]).toDF("text")

model = jsl_ner_pipeline.fit(df)
res = model.transform(df)

```

</div>

## Results

```bash
+------------------------------------------------+-----+
|ner_chunk                                       |label|
+------------------------------------------------+-----+
|SILICIUM DE PROVENCE S.A.S                      |ORG  |
|France                                          |LOC  |
|Usine de Saint Auban                            |LOC  |
|France                                          |LOC  |
|Mr.Frank Wouters                                |PER  |
|Borrower                                        |PER  |
|EVERGREEN SOLAR INC                             |ORG  |
|Delaware                                        |LOC  |
|U.S.A                                           |LOC  |
|Bartlett Street                                 |LOC  |
|Marlboro                                        |LOC  |
|Massachusetts                                   |LOC  |
|U.S.A                                           |LOC  |
|Richard Chleboski                               |PER  |
|Lender                                          |PER  |
+------------------------------------------------+-----+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finner_sec_conll|
|Type:|finance|
|Compatibility:|Spark NLP for Finance 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|16.3 MB|

## References

Manual annotations, coNll2003 and financial documents obtained from U.S. Security and Exchange Commission (SEC) filings.

## Benchmarking

```bash
label	       tp	 fp	 fn	 prec	     rec	     f1
B-LOC	       14	 6	 11	 0.7	     0.56	     0.6222222
I-ORG	       59	 30	 3	 0.66292137	 0.9516129	 0.781457
I-LOC	       32	 2	 22	 0.9411765	 0.5925926	 0.7272727
I-PER	       18	 4	 5	 0.8181818	 0.7826087	 0.8
B-ORG	       47	 17	 5	 0.734375	 0.90384614	 0.8103449
B-PER	       211	 7	 2	 0.9678899	 0.9906103	 0.97911835
Macro-average  381   66  48  0.804091    0.7968784   0.8004684
Micro-average  381   66  48  0.852349    0.8881119   0.869863
```