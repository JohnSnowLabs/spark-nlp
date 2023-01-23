{%- capture title -%}
AssertionChunkConverter
{%- endcapture -%}

{%- capture model -%}
model
{%- endcapture -%}

{%- capture model_description -%}

This annotator creates a `CHUNK` column with metadata useful for training an Assertion Status Detection model (see [AssertionDL](https://nlp.johnsnowlabs.com/docs/en/licensed_annotators#assertiondl)).

In some cases, there may be issues while creating the chunk column when using token indices that can lead to loss of data to train assertion status models.

The `AssertionChunkConverter` annotator uses both begin and end indices of the tokens as input to add a more robust metadata to the chunk column in a way that improves the reliability of the indices and avoid loss of data.

> *NOTE*: Chunk begin and end indices in the assertion status model training dataframe can be populated using the new version of ALAB module.

{%- endcapture -%}

{%- capture model_input_anno -%}
TOKEN
{%- endcapture -%}

{%- capture model_output_anno -%}
CHUNK
{%- endcapture -%}

{%- capture model_python_medical -%}

data = spark.createDataFrame(
    [
        [
            "An angiography showed bleeding in two vessels off of the Minnie supplying the sigmoid that were succesfully embolized.",
            "Minnie",
            57,
            64,
        ],
        [
            "After discussing this with his PCP, Leon was clear that the patient had had recurrent DVTs and ultimately a PE and his PCP felt strongly that he required long-term anticoagulation ",
            "PCP",
            31,
            34,
        ],
    ]
).toDF("text", "target", "char_begin", "char_end")

document_assembler = DocumentAssembler().setInputCol("text").setOutputCol("document")

sentenceDetector = (
    SentenceDetector().setInputCols(["document"]).setOutputCol("sentence")
)

tokenizer = Tokenizer().setInputCols(["sentence"]).setOutputCol("tokens")

converter = (
    AssertionChunkConverter()
    .setInputCols("tokens")
    .setChunkTextCol("target")
    .setChunkBeginCol("char_begin")
    .setChunkEndCol("char_end")
    .setOutputTokenBeginCol("token_begin")
    .setOutputTokenEndCol("token_end")
    .setOutputCol("chunk")
)

pipeline = Pipeline().setStages(
    [document_assembler, sentenceDetector, tokenizer, converter]
)

results = pipeline.fit(data).transform(data)

results.selectExpr(
    "target",
    "char_begin",
    "char_end",
    "token_begin",
    "token_end",
    "tokens[token_begin].result",
    "tokens[token_end].result",
    "target",
    "chunk",
).show(truncate=False)
+------+----------+--------+-----------+---------+--------------------------+------------------------+------+----------------------------------------------+
|target|char_begin|char_end|token_begin|token_end|tokens[token_begin].result|tokens[token_end].result|target|chunk                                         |
+------+----------+--------+-----------+---------+--------------------------+------------------------+------+----------------------------------------------+
|Minnie|57        |64      |10         |10       |Minnie                    |Minnie                  |Minnie|[{chunk, 57, 62, Minnie, {sentence -> 0}, []}]|
|PCP   |31        |34      |5          |5        |PCP                       |PCP                     |PCP   |[{chunk, 31, 33, PCP, {sentence -> 0}, []}]   |
+------+----------+--------+-----------+---------+--------------------------+------------------------+------+----------------------------------------------+

{%- endcapture -%}

{%- capture model_api_link -%}
[AssertionChunkConverter](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/assertion/AssertionChunkConverter.html)
{%- endcapture -%}

{%- capture model_python_api_link -%}
[AssertionChunkConverter](https://nlp.johnsnowlabs.com/licensed/api/python/reference/autosummary/sparknlp_jsl/annotator/assertion/assertion_chunk_converter/index.html#sparknlp_jsl.annotator.assertion.assertion_chunk_converter.AssertionChunkConverter)
{%- endcapture -%}


{% include templates/licensed_approach_model_medical_fin_leg_template.md
title=title
model=model
approach=approach
model_description=model_description
model_input_anno=model_input_anno
model_output_anno=model_output_anno
model_python_medical=model_python_medical
model_api_link=model_api_link
model_python_api_link=model_python_api_link
%}
