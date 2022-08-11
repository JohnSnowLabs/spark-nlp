---
layout: model
title: Legal NER (Parties, Dates, Document Type)
author: John Snow Labs
name: legner_contract_doc_parties
date: 2022-08-11
tags: [en, legal, ner, agreements, licensed]
task: Named Entity Recognition
language: en
edition: Spark NLP for Legal 1.0.0
spark_version: 3.2
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a Legal NER Model, aimed to process the first page of the agreements when information can be found about:
- Parties of the contract/agreement;
- Aliases of those parties, or how those parties will be called further on in the document;
- Document Type;
- Effective Date of the agreement;

This model can be used all along with its Relation Extraction model to retrieve the relations between these entities.

Other models can be found to detect other parts of the document, as Headers/Subheaders, Signers, "Will-do", etc.

## Predicted Entities

`PARTY`, `EFFDATE`, `DOC`, `ALIA`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/finance/LEGALNER_PARTIES/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legner_contract_doc_parties_en_1.0.0_3.2_1660212978025.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler()\
        .setInputCol("text")\
        .setOutputCol("document")
        
sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl","xx")\
        .setInputCols(["document"])\
        .setOutputCol("sentence")

tokenizer = Tokenizer()\
        .setInputCols(["sentence"])\
        .setOutputCol("token")

embeddings = RoBertaEmbeddings.pretrained("roberta_embeddings_legal_roberta_base", "en") \
        .setInputCols("sentence", "token") \
        .setOutputCol("embeddings")\

ner_model = LegalNerModel().pretrained('legner_contract_doc_parties', 'en', 'legal/models')\
        .setInputCols(["sentence", "token", "embeddings"])\
        .setOutputCol("ner")

ner_converter = NerConverter()\
        .setInputCols(["sentence","token","ner"])\
        .setOutputCol("ner_chunk")

nlpPipeline = Pipeline(stages=[
        documentAssembler,
        sentenceDetector,
        tokenizer,
        embeddings,
        ner_model,
        ner_converter])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = nlpPipeline.fit(empty_data)

text = ["""
INTELLECTUAL PROPERTY AGREEMENT

This INTELLECTUAL PROPERTY AGREEMENT (this "Agreement"), dated as of December 31, 2018 (the "Effective Date") is entered into by and between Armstrong Flooring, Inc., a Delaware corporation ("Seller") and AFI Licensing LLC, a Delaware limited liability company ("Licensing" and together with Seller, "Arizona") and AHF Holding, Inc. (formerly known as Tarzan HoldCo, Inc.), a Delaware corporation ("Buyer") and Armstrong Hardwood Flooring Company, a Tennessee corporation (the "Company" and together with Buyer the "Buyer Entities") (each of Arizona on the one hand and the Buyer Entities on the other hand, a "Party" and collectively, the "Parties").
"""]

res = model.transform(spark.createDataFrame([text]).toDF("text"))
```

</div>

## Results

```bash
+------------+---------+
|       token|ner_label|
+------------+---------+
|INTELLECTUAL|    B-DOC|
|    PROPERTY|    I-DOC|
|   AGREEMENT|    I-DOC|
|        This|        O|
|INTELLECTUAL|    B-DOC|
|    PROPERTY|    I-DOC|
|   AGREEMENT|    I-DOC|
|           (|        O|
|        this|        O|
|           "|        O|
|   Agreement|        O|
|         "),|        O|
|       dated|        O|
|          as|        O|
|          of|        O|
|    December|B-EFFDATE|
|          31|I-EFFDATE|
|           ,|I-EFFDATE|
|        2018|I-EFFDATE|
|           (|        O|
|         the|        O|
|           "|        O|
|   Effective|        O|
|        Date|        O|
|          ")|        O|
|          is|        O|
|     entered|        O|
|        into|        O|
|          by|        O|
|         and|        O|
|     between|        O|
|   Armstrong|  B-PARTY|
|    Flooring|  I-PARTY|
|           ,|  I-PARTY|
|         Inc|  I-PARTY|
|          .,|        O|
|           a|        O|
|    Delaware|        O|
| corporation|        O|
|          ("|        O|
|      Seller|  B-ALIAS|
|          ")|        O|
|         and|        O|
|         AFI|  B-PARTY|
|   Licensing|  I-PARTY|
|         LLC|  I-PARTY|
|           ,|        O|
|           a|        O|
|    Delaware|        O|
|     limited|        O|
|   liability|        O|
|     company|        O|
|          ("|        O|
|   Licensing|  B-ALIAS|
|           "|        O|
|         and|        O|
|    together|        O|
|        with|        O|
|      Seller|  B-ALIAS|
|           ,|        O|
|           "|        O|
|     Arizona|  B-ALIAS|
|          ")|        O|
|         and|        O|
|         AHF|  B-PARTY|
|     Holding|  I-PARTY|
|           ,|  I-PARTY|
|         Inc|  I-PARTY|
|           .|        O|
|           (|        O|
|    formerly|        O|
|       known|        O|
|          as|        O|
|      Tarzan|        O|
|      HoldCo|        O|
|           ,|        O|
|         Inc|        O|
|         .),|        O|
|           a|        O|
|    Delaware|        O|
| corporation|        O|
|          ("|        O|
|       Buyer|  B-ALIAS|
|          ")|        O|
|         and|        O|
|   Armstrong|  B-PARTY|
|    Hardwood|  I-PARTY|
|    Flooring|  I-PARTY|
|     Company|  I-PARTY|
------------------------
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legner_contract_doc_parties|
|Type:|legal|
|Compatibility:|Spark NLP for Legal 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|16.3 MB|

## References

Manual annotations on CUAD dataset

## Benchmarking

```bash
label	 tp	 fp	 fn	 prec	 rec	 f1
I-PARTY	 216	 28	 37	 0.8852459	 0.85375494	 0.8692153
B-EFFDATE	 11	 2	 2	 0.84615386	 0.84615386	 0.84615386
B-DOC	 20	 4	 8	 0.8333333	 0.71428573	 0.7692307
I-EFFDATE	 50	 6	 2	 0.89285713	 0.96153843	 0.9259259
I-ALIAS	 4	 1	 2	 0.8	 0.6666667	 0.72727275
I-DOC	 57	 5	 9	 0.91935486	 0.8636364	 0.89062506
B-PARTY	 55	 15	 24	 0.78571427	 0.6962025	 0.738255
B-ALIAS	 55	 21	 9	 0.7236842	 0.859375	 0.78571427
```