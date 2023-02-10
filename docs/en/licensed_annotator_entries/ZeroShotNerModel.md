{%- capture title -%}
ZeroShotNerModel
{%- endcapture -%}

{%- capture model -%}
model
{%- endcapture -%}

{%- capture model_description -%}

This is a zero shot named entity recognition based on `RoBertaForQuestionAnswering`. Zero shot models excel at generalization, meaning that the model can accurately predict entities in very different data sets without the need to fine tune the model or train from scratch for each different domain.

Even though a model trained to solve a specific problem can achieve better accuracy than a zero-shot model in this specific task, it probably won't be be useful in a different task. That is where zero-shot models shows its usefulness by being able to achieve good results in many different scenarions.

{%- endcapture -%}

{%- capture model_input_anno -%}
DOCUMENT, TOKEN 
{%- endcapture -%}

{%- capture model_output_anno -%}
NAMED_ENTITY
{%- endcapture -%}

{%- capture model_python_medical -%}

documentAssembler = DocumentAssembler().setInputCol("text").setOutputCol("document")

sentenceDetector = (
    SentenceDetector().setInputCols(["document"]).setOutputCol("sentence")
)

tokenizer = Tokenizer().setInputCols(["sentence"]).setOutputCol("token")

zero_shot_ner = (
    ZeroShotNerModel.pretrained("zero_shot_ner_roberta", "en", "clinical/models")
    .setEntityDefinitions(
        {
            "PROBLEM": [
                "What is the disease?",
                "What is his symptom?",
                "What is her disease?",
                "What is his disease?",
                "What is the problem?",
                "What does a patient suffer",
                "What was the reason that the patient is admitted to the clinic?",
            ],
            "DRUG": [
                "Which drug?",
                "Which is the drug?",
                "What is the drug?",
                "Which drug does he use?",
                "Which drug does she use?",
                "Which drug do I use?",
                "Which drug is prescribed for a symptom?",
            ],
            "ADMISSION_DATE": ["When did patient admitted to a clinic?"],
            "PATIENT_AGE": [
                "How old is the patient?",
                "What is the gae of the patient?",
            ],
        }
    )
    .setInputCols(["sentence", "token"])
    .setOutputCol("zero_shot_ner")
    .setPredictionThreshold(0.1)
)  # default 0.01

ner_converter = (
    sparknlp.annotators.NerConverter()
    .setInputCols(["sentence", "token", "zero_shot_ner"])
    .setOutputCol("ner_chunk")
)
pipeline = Pipeline(
    stages=[
        documentAssembler,
        sentenceDetector,
        tokenizer,
        zero_shot_ner,
        ner_converter,
    ]
)

zero_shot_ner_model = pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

text_list = [
    "The doctor pescribed Majezik for my severe headache.",
    "The patient was admitted to the hospital for his colon cancer.",
    "27 years old patient was admitted to clinic on Sep 1st by Dr. X for a right-sided pleural effusion for thoracentesis.",
]

data = spark.createDataFrame(text_list, StringType()).toDF("text")

results = zero_shot_ner_model.transform(data)
results.select(
    F.explode(
        F.arrays_zip(
            results.token.result,
            results.zero_shot_ner.result,
            results.zero_shot_ner.metadata,
            results.zero_shot_ner.begin,
            results.zero_shot_ner.end,
        )
    ).alias("cols")
).select(
    F.expr("cols['0']").alias("token"),
    F.expr("cols['1']").alias("ner_label"),
    F.expr("cols['2']['sentence']").alias("sentence"),
    F.expr("cols['3']").alias("begin"),
    F.expr("cols['4']").alias("end"),
    F.expr("cols['2']['confidence']").alias("confidence"),
).show(
    50, truncate=100
)
+-------------+----------------+--------+-----+---+----------+
        token|       ner_label|sentence|begin|end|confidence|
+-------------+----------------+--------+-----+---+----------+
          The|               O|       0|    0|  2|      null|
       doctor|               O|       0|    4|  9|      null|
    pescribed|               O|       0|   11| 19|      null|
      Majezik|          B-DRUG|       0|   21| 27| 0.6467137|
          for|               O|       0|   29| 31|      null|
           my|               O|       0|   33| 34|      null|
       severe|       B-PROBLEM|       0|   36| 41|0.55263567|
     headache|       I-PROBLEM|       0|   43| 50|0.55263567|
            .|               O|       0|   51| 51|      null|
          The|               O|       0|    0|  2|      null|
      patient|               O|       0|    4| 10|      null|
          was|               O|       0|   12| 14|      null|
     admitted|               O|       0|   16| 23|      null|
           to|               O|       0|   25| 26|      null|
          the|               O|       0|   28| 30|      null|
     hospital|               O|       0|   32| 39|      null|
          for|               O|       0|   41| 43|      null|
          his|               O|       0|   45| 47|      null|
        colon|       B-PROBLEM|       0|   49| 53| 0.8898501|
       cancer|       I-PROBLEM|       0|   55| 60| 0.8898501|
            .|               O|       0|   61| 61|      null|
           27|   B-PATIENT_AGE|       0|    0|  1| 0.6943086|
        years|   I-PATIENT_AGE|       0|    3|  7| 0.6943086|
          old|   I-PATIENT_AGE|       0|    9| 11| 0.6943086|
      patient|               O|       0|   13| 19|      null|
          was|               O|       0|   21| 23|      null|
     admitted|               O|       0|   25| 32|      null|
           to|               O|       0|   34| 35|      null|
       clinic|               O|       0|   37| 42|      null|
           on|               O|       0|   44| 45|      null|
          Sep|B-ADMISSION_DATE|       0|   47| 49|0.95646083|
          1st|I-ADMISSION_DATE|       0|   51| 53|0.95646083|
           by|               O|       0|   55| 56|      null|
           Dr|               O|       0|   58| 59|      null|
            .|               O|       0|   60| 60|      null|
            X|               O|       0|   62| 62|      null|
          for|               O|       0|   64| 66|      null|
            a|       B-PROBLEM|       0|   68| 68|0.50026655|
  right-sided|       I-PROBLEM|       0|   70| 80|0.50026655|
      pleural|       I-PROBLEM|       0|   82| 88|0.50026655|
     effusion|       I-PROBLEM|       0|   90| 97|0.50026655|
          for|       I-PROBLEM|       0|   99|101|0.50026655|
thoracentesis|       I-PROBLEM|       0|  103|115|0.50026655|
            .|               O|       0|  116|116|      null|
+-------------+----------------+--------+-----+---+----------+

{%- endcapture -%}

{%- capture model_python_finance -%}

document_assembler = (
    nlp.DocumentAssembler().setInputCol("text").setOutputCol("document")
)

sentence_detector = (
    nlp.SentenceDetector().setInputCols(["document"]).setOutputCol("sentence")
)

tokenizer = nlp.Tokenizer().setInputCols("sentence").setOutputCol("token")

zero_shot_ner = (
    finance.ZeroShotNerModel.pretrained(
        "finner_roberta_zeroshot", "en", "finance/models"
    )
    .setInputCols(["sentence", "token"])
    .setOutputCol("zero_shot_ner")
    .setEntityDefinitions(
        {
            "DATE": [
                "When was the company acquisition?",
                "When was the company purchase agreement?",
            ],
            "ORG": ["Which company was acquired?"],
            "PRODUCT": ["Which product?"],
            "PROFIT_INCREASE": ["How much has the gross profit increased?"],
            "REVENUES_DECLINED": ["How much has the revenues declined?"],
            "OPERATING_LOSS_2020": ["Which was the operating loss in 2020"],
            "OPERATING_LOSS_2019": ["Which was the operating loss in 2019"],
        }
    )
)

ner_converter = (
    nlp.NerConverter()
    .setInputCols(["sentence", "token", "zero_shot_ner"])
    .setOutputCol("ner_chunk")
)

pipeline = nlp.Pipeline(
    stages=[
        document_assembler,
        sentence_detector,
        tokenizer,
        zero_shot_ner,
        ner_converter,
    ]
)


from pyspark.sql.types import StringType

sample_text = [
    "In March 2012, as part of a longer-term strategy, the Company acquired Vertro, Inc., which owned and operated the ALOT product portfolio.",
    "In February 2017, the Company entered into an asset purchase agreement with NetSeer, Inc.",
    "While our gross profit margin increased to 81.4% in 2020 from 63.1% in 2019, our revenues declined approximately 27% in 2020 as compared to 2019.",
    "We reported an operating loss of approximately $8,048,581 million in 2020 as compared to an operating loss of $7,738,193 in 2019.",
]

p_model = pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

res = p_model.transform(spark.createDataFrame(sample_text, StringType()).toDF("text"))

res.select(
    F.explode(
        F.arrays_zip(
            res.ner_chunk.result,
            res.ner_chunk.begin,
            res.ner_chunk.end,
            res.ner_chunk.metadata,
        )
    ).alias("cols")
).select(
    F.expr("cols['0']").alias("chunk"), F.expr("cols['3']['entity']").alias("ner_label")
).filter(
    "ner_label!='O'"
).show(
    truncate=False
)
+------------------+-------------------+
|chunk             |ner_label          |
+------------------+-------------------+
|March 2012        |DATE               |
|Vertro            |ORG                |
|ALOT              |PRODUCT            |
|February 2017     |DATE               |
|NetSeer           |ORG                |
|81.4%             |PROFIT_INCREASE    |
|27%               |REVENUES_DECLINED  |
|$8,048,581 million|OPERATING_LOSS_2020|
|$7,738,193        |OPERATING_LOSS_2019|
|2019              |DATE               |
+------------------+-------------------+

{%- endcapture -%}

{%- capture model_python_legal -%}

documentAssembler = nlp.DocumentAssembler().setInputCol("text").setOutputCol("document")

sentence = nlp.SentenceDetector().setInputCols(["document"]).setOutputCol("sentence")

tokenizer = nlp.Tokenizer().setInputCols("sentence").setOutputCol("token")

zero_shot_ner = (
    legal.ZeroShotNerModel.pretrained("legner_roberta_zeroshot", "en", "legal/models")
    .setInputCols(["sentence", "token"])
    .setOutputCol("zero_shot_ner")
    .setEntityDefinitions(
        {
            "DATE": [
                "When was the company acquisition?",
                "When was the company purchase agreement?",
                "When was the agreement?",
            ],
            "ORG": ["Which company?"],
            "STATE": ["Which state?"],
            "AGREEMENT": ["What kind of agreement?"],
            "LICENSE": ["What kind of license?"],
            "LICENSE_RECIPIENT": ["To whom the license is granted?"],
        }
    )
)


nerconverter = (
    nlp.NerConverter()
    .setInputCols(["sentence", "token", "zero_shot_ner"])
    .setOutputCol("ner_chunk")
)

pipeline = nlp.Pipeline(
    stages=[
        documentAssembler,
        sentence,
        tokenizer,
        zero_shot_ner,
        nerconverter,
    ]
)

from pyspark.sql.types import StructType, StructField, StringType

sample_text = [
    "In March 2012, as part of a longer-term strategy, the Company acquired Vertro, Inc., which owned and operated the ALOT product portfolio.",
    "In February 2017, the Company entered into an asset purchase agreement with NetSeer, Inc.",
    "This INTELLECTUAL PROPERTY AGREEMENT, dated as of December 31, 2018 (the 'Effective Date') is entered into by and between Armstrong Flooring, Inc., a Delaware corporation ('Seller') and AFI Licensing LLC, a Delaware company (the 'Licensee')",
    "The Company hereby grants to Seller a perpetual, non- exclusive, royalty-free license",
]

p_model = pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

res = p_model.transform(spark.createDataFrame(sample_text, StringType()).toDF("text"))

res.select(
    F.explode(
        F.arrays_zip(
            res.ner_chunk.result,
            res.ner_chunk.begin,
            res.ner_chunk.end,
            res.ner_chunk.metadata,
        )
    ).alias("cols")
).select(
    F.expr("cols['0']").alias("chunk"), F.expr("cols['3']['entity']").alias("ner_label")
).filter(
    "ner_label!='O'"
).show(
    truncate=False
)
+-------------------------------------+-----------------+
|chunk                                |ner_label        |
+-------------------------------------+-----------------+
|March 2012                           |DATE             |
|Vertro, Inc                          |ORG              |
|February 2017                        |DATE             |
|asset purchase agreement             |AGREEMENT        |
|NetSeer                              |ORG              |
|INTELLECTUAL PROPERTY                |AGREEMENT        |
|December 31, 2018                    |DATE             |
|Armstrong Flooring                   |LICENSE_RECIPIENT|
|Delaware                             |STATE            |
|AFI Licensing LLC, a Delaware company|LICENSE_RECIPIENT|
|Seller                               |LICENSE_RECIPIENT|
|perpetual                            |LICENSE          |
|non- exclusive                       |LICENSE          |
|royalty-free                         |LICENSE          |
+-------------------------------------+-----------------+

{%- endcapture -%}


{%- capture model_scala_medical -%}

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentenceDetector = new SentenceDetector()
  .setInputCols(Array("document"))
  .setOutputCol("sentences")

val zeroShotNer = ZeroShotNerModel
  .pretrained()
  .setEntityDefinitions(
    Map(
      "NAME" -> Array("What is his name?", "What is her name?"),
      "CITY" -> Array("Which city?")))
  .setPredictionThreshold(0.01f)
  .setInputCols("sentences")
  .setOutputCol("zero_shot_ner")

val pipeline = new Pipeline()
  .setStages(Array(
    documentAssembler,
    sentenceDetector,
    zeroShotNer))

val model = pipeline.fit(Seq("").toDS.toDF("text"))
val results = model.transform(
  Seq("Clara often travels between New York and Paris.").toDS.toDF("text"))

results
  .selectExpr("document", "explode(zero_shot_ner) AS entity")
  .select(
    col("entity.result"),
    col("entity.metadata.word"),
    col("entity.metadata.sentence"),
    col("entity.begin"),
    col("entity.end"),
    col("entity.metadata.confidence"),
    col("entity.metadata.question"))
  .show(truncate=false)

+------+-----+--------+-----+---+----------+------------------+
|result|word |sentence|begin|end|confidence|question          |
+------+-----+--------+-----+---+----------+------------------+
|B-CITY|Paris|0       |41   |45 |0.78655756|Which is the city?|
|B-CITY|New  |0       |28   |30 |0.29346612|Which city?       |
|I-CITY|York |0       |32   |35 |0.29346612|Which city?       |
+------+-----+--------+-----+---+----------+------------------+

{%- endcapture -%}


{%- capture model_scala_finance -%}

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentenceDetector = new SentenceDetector()
  .setInputCols(Array("document"))
  .setOutputCol("sentences")

val zeroShotNer = ZeroShotNerModel
  .pretrained()
  .setEntityDefinitions(
    Map(
      "NAME" -> Array("What is his name?", "What is her name?"),
      "CITY" -> Array("Which city?")))
  .setPredictionThreshold(0.01f)
  .setInputCols("sentences")
  .setOutputCol("zero_shot_ner")

val pipeline = new Pipeline()
  .setStages(Array(
    documentAssembler,
    sentenceDetector,
    zeroShotNer))

val model = pipeline.fit(Seq("").toDS.toDF("text"))
val results = model.transform(
  Seq("Clara often travels between New York and Paris.").toDS.toDF("text"))

results
  .selectExpr("document", "explode(zero_shot_ner) AS entity")
  .select(
    col("entity.result"),
    col("entity.metadata.word"),
    col("entity.metadata.sentence"),
    col("entity.begin"),
    col("entity.end"),
    col("entity.metadata.confidence"),
    col("entity.metadata.question"))
  .show(truncate=false)

+------+-----+--------+-----+---+----------+------------------+
|result|word |sentence|begin|end|confidence|question          |
+------+-----+--------+-----+---+----------+------------------+
|B-CITY|Paris|0       |41   |45 |0.78655756|Which is the city?|
|B-CITY|New  |0       |28   |30 |0.29346612|Which city?       |
|I-CITY|York |0       |32   |35 |0.29346612|Which city?       |
+------+-----+--------+-----+---+----------+------------------+

{%- endcapture -%}


{%- capture model_scala_legal -%}

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentenceDetector = new SentenceDetector()
  .setInputCols(Array("document"))
  .setOutputCol("sentences")

val zeroShotNer = ZeroShotNerModel
  .pretrained()
  .setEntityDefinitions(
    Map(
      "NAME" -> Array("What is his name?", "What is her name?"),
      "CITY" -> Array("Which city?")))
  .setPredictionThreshold(0.01f)
  .setInputCols("sentences")
  .setOutputCol("zero_shot_ner")

val pipeline = new Pipeline()
  .setStages(Array(
    documentAssembler,
    sentenceDetector,
    zeroShotNer))

val model = pipeline.fit(Seq("").toDS.toDF("text"))
val results = model.transform(
  Seq("Clara often travels between New York and Paris.").toDS.toDF("text"))

results
  .selectExpr("document", "explode(zero_shot_ner) AS entity")
  .select(
    col("entity.result"),
    col("entity.metadata.word"),
    col("entity.metadata.sentence"),
    col("entity.begin"),
    col("entity.end"),
    col("entity.metadata.confidence"),
    col("entity.metadata.question"))
  .show(truncate=false)

+------+-----+--------+-----+---+----------+------------------+
|result|word |sentence|begin|end|confidence|question          |
+------+-----+--------+-----+---+----------+------------------+
|B-CITY|Paris|0       |41   |45 |0.78655756|Which is the city?|
|B-CITY|New  |0       |28   |30 |0.29346612|Which city?       |
|I-CITY|York |0       |32   |35 |0.29346612|Which city?       |
+------+-----+--------+-----+---+----------+------------------+

{%- endcapture -%}

{%- capture model_api_link -%}
[ZeroShotNerModel](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/finance/token_classification/ner/ZeroShotNerModel.html)
{%- endcapture -%}

{%- capture model_python_api_link -%}
[ZeroShotNerModel](https://nlp.johnsnowlabs.com/licensed/api/python/reference/autosummary/sparknlp_jsl/annotator/ner/zero_shot_ner/index.html#sparknlp_jsl.annotator.ner.zero_shot_ner.ZeroShotNerModel)
{%- endcapture -%}


{% include templates/licensed_approach_model_medical_fin_leg_template.md
title=title
model=model
model_description=model_description
model_input_anno=model_input_anno
model_output_anno=model_output_anno
model_python_medical=model_python_medical
model_python_finance=model_python_finance
model_python_legal=model_python_legal
model_scala_legal=model_scala_legal
model_scala_finance=model_scala_finance
model_api_link=model_api_link
model_python_api_link=model_python_api_link
%}
