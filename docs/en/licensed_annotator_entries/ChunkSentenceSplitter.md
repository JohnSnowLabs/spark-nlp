{%- capture title -%}
ChunkSentenceSplitter
{%- endcapture -%}

{%- capture model -%}
model
{%- endcapture -%}

{%- capture model_description -%}

`ChunkSentenceSplitter` annotator can split the documents into chunks according to separators given as `CHUNK` columns. It is useful when you need to perform different models or analysis in different sections of your document (for example, for different headers, clauses, items, etc.). The given separator chunk can be the output from, for example, [RegexMatcher](https://nlp.johnsnowlabs.com/docs/en/annotators#regexmatcher) or [NerModel](https://nlp.johnsnowlabs.com/docs/en/licensed_annotators#nermodel).

For detailed usage of this annotator, visit [this notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/18.Chunk_Sentence_Splitter.ipynb) from our `Spark NLP Workshop`.

{%- endcapture -%}

{%- capture model_input_anno -%}
DOCUMENT, CHUNK
{%- endcapture -%}

{%- capture model_output_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture model_python_medical -%}

# Defining the pipeline

documentAssembler = DocumentAssembler().setInputCol("text").setOutputCol("document")

tokenizer = Tokenizer().setInputCols(["document"]).setOutputCol("token")

tokenClassifier = (
    MedicalBertForTokenClassifier.pretrained(
        "bert_token_classifier_ner_jsl_slim", "en", "clinical/models"
    )
    .setInputCols("token", "document")
    .setOutputCol("ner")
    .setCaseSensitive(True)
)

ner_converter = (
    NerConverter()
    .setInputCols(["document", "token", "ner"])
    .setOutputCol("ner_chunk")
    .setWhiteList(["Header"])
)

chunkSentenceSplitter = (
    ChunkSentenceSplitter()
    .setInputCols("document", "ner_chunk")
    .setOutputCol("paragraphs")
    .setGroupBySentences(False)
)

pipeline = Pipeline(
    stages=[
        documentAssembler,
        tokenizer,
        tokenClassifier,
        ner_converter,
        chunkSentenceSplitter,
    ]
)

empty_df = spark.createDataFrame([[""]]).toDF("text")
pipeline_model = pipeline.fit(empty_df)

sentences = [
    [
        """ADMISSION DIAGNOSIS Right pleural effusion and suspected malignant mesothelioma.
        PRINCIPAL DIAGNOSIS Right pleural effusion, suspected malignant mesothelioma.
        REVIEW OF SYSTEMS Right pleural effusion, firm nodules, diffuse scattered throughout the right pleura and diaphragmatic surface.
    """
    ]
]

df = spark.createDataFrame(sentences).toDF("text")

paragraphs = pipeline_model.transform(df)
 paragraphs.selectExpr("explode(paragraphs) as result").selectExpr("result.result","result.metadata.entity", "result.metadata.splitter_chunk").show(truncate=80)
+--------------------------------------------------------------------------------+------+-------------------+
|                                                                          result|entity|     splitter_chunk|
+--------------------------------------------------------------------------------+------+-------------------+
|ADMISSION DIAGNOSIS Right pleural effusion and suspected malignant mesothelio...|Header|ADMISSION DIAGNOSIS|
|PRINCIPAL DIAGNOSIS Right pleural effusion, suspected malignant mesothelioma....|Header|PRINCIPAL DIAGNOSIS|
|REVIEW OF SYSTEMS Right pleural effusion, firm nodules, diffuse scattered thr...|Header|  REVIEW OF SYSTEMS|
+--------------------------------------------------------------------------------+------+-------------------+

{%- endcapture -%}

{%- capture model_scala_medical -%}

val data = Seq(text,text).toDS.toDF("text")
val documentAssembler = new DocumentAssembler().setInputCol("text").setOutputCol("doc")
val regexMatcher = new RegexMatcher().setInputCols("doc").setOutputCol("chunks").setExternalRules("src/test/resources/chunker/title_regex.txt",",")
val chunkSentenceSplitter = new ChunkSentenceSplitter().setInputCols("chunks","doc").setOutputCol("paragraphs")
val pipeline =  new Pipeline().setStages(Array(documentAssembler,regexMatcher,chunkSentenceSplitter))
val result = pipeline.fit(data).transform(data).select("paragraphs")
result.show(truncate = false)

{%- endcapture -%}

{%- capture model_api_link -%}
[ChunkSentenceSplitter](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/chunker/ChunkSentenceSplitter.html)
{%- endcapture -%}

{%- capture model_python_api_link -%}
[ChunkSentenceSplitter](https://nlp.johnsnowlabs.com/licensed/api/python/reference/autosummary/sparknlp_jsl/annotator/chunker/chunk_sentence_splitter/index.html#sparknlp_jsl.annotator.chunker.chunk_sentence_splitter.ChunkSentenceSplitter)
{%- endcapture -%}


{% include templates/licensed_approach_model_medical_fin_leg_template.md
title=title
model=model
approach=approach
model_description=model_description
model_input_anno=model_input_anno
model_output_anno=model_output_anno
model_python_medical=model_python_medical
model_scala_medical=model_scala_medical
model_api_link=model_api_link
model_python_api_link=model_python_api_link
%}
