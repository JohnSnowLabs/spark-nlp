{%- capture title -%}
ChunkConverter
{%- endcapture -%}

{%- capture model -%}
model
{%- endcapture -%}

{%- capture model_description -%}
Convert chunks from [RegexMatcher](https://nlp.johnsnowlabs.com/docs/en/annotators#regexmatcher) to chunks with a entity in the metadata.

This annotator is important when the user wants to merge entities identified by NER models together with rules-based matching used by the RegexMathcer annotator. In the following steps of the pipeline, all the identified entities can be treated in a unified field.

{%- endcapture -%}

{%- capture model_input_anno -%}
DOCUMENT, CHUNK
{%- endcapture -%}

{%- capture model_output_anno -%}
CHUNK
{%- endcapture -%}

{%- capture model_python_medical -%}
# Creating the pipeline

documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentenceDetector = SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")

ner_model = MedicalNerModel.pretrained("ner_clinical_large","en","clinical/models") \
    .setInputCols("sentence","token","embeddings") \
    .setOutputCol("ner")

ner_converter= NerConverter()\
    .setInputCols(["sentence", "token", "ner"])\
    .setOutputCol("ner_chunk")\

regex_matcher = RegexMatcher()\
    .setInputCols('document')\
    .setStrategy("MATCH_ALL")\
    .setOutputCol("regex_matches")\
    .setExternalRules(path='file:/dbfs/regex_rules.txt', delimiter=',')

chunkConverter = ChunkConverter()\
    .setInputCols("regex_matches")\
    .setOutputCol("regex_chunk")

merger= ChunkMergeApproach()\
    .setInputCols(["regex_chunk", "ner_chunk"])\
    .setOutputCol("merged_chunks")\
    .setMergeOverlapping(True)\
    .setChunkPrecedence("field")

pipeline= Pipeline(stages=[
                           documentAssembler,
                           sentenceDetector,
                           tokenizer,
                           word_embeddings,
                           ner_model,
                           ner_converter,
                           regex_matcher,
                           chunkConverter,
                           merger
])

empty_df= spark.createDataFrame([[""]]).toDF("text")
model= pipeline.fit(empty_df)

lp_model = LightPipeline(model)
results = lp_model.fullAnnotate(sample_text)[0]

# Displaying the results

chunk= []
merge= []
for result in list(results["merged_chunks"]):
  merge.append(result.metadata["entity"])
  chunk.append(result.result)
df_merge = pd.DataFrame({"chunk": chunk,  "merged_entity": merge})
df_merge

|                                          chunk |  merged_entity |
|-----------------------------------------------:|---------------:|
|                       POSTOPERATIVE DIAGNOSIS: | SECTION_HEADER |
|                       Cervical lymphadenopathy |        PROBLEM |
|                                     PROCEDURE: | SECTION_HEADER |
| Excisional biopsy of right cervical lymph node |           TEST |
|                                    ANESTHESIA: | SECTION_HEADER |
|                General endotracheal anesthesia |      TREATMENT |
|                      Right cervical lymph node |        PROBLEM |
|                                           EBL: | SECTION_HEADER |
|                                 COMPLICATIONS: | SECTION_HEADER |
|                                      FINDINGS: | SECTION_HEADER |
|                    Enlarged level 2 lymph node |        PROBLEM |
| ...                                            |                |

{%- endcapture -%}

{%- capture model_scala_medical -%}
val sampleDataset = ResourceHelper.spark.createDataFrame(Seq(
 (1, "My first sentence with the first rule. This is my second sentence with ceremonies rule.")
)).toDF("id", "text")

val documentAssembler = new DocumentAssembler().setInputCol("text").setOutputCol("document")

val sentence = new SentenceDetector().setInputCols("document").setOutputCol("sentence")

val regexMatcher = new RegexMatcher()
 .setExternalRules(ExternalResource("src/test/resources/regex-matcher/rules.txt", ReadAs.TEXT, Map("delimiter" -> ",")))
 .setInputCols(Array("sentence"))
 .setOutputCol("regex")
 .setStrategy(strategy)

val chunkConverter = new ChunkConverter().setInputCols("regex").setOutputCol("chunk")

val pipeline = new Pipeline().setStages(Array(documentAssembler, sentence, regexMatcher,chunkConverter))

val results = pipeline.fit(sampleDataset).transform(sampleDataset)
results.select("chunk").show(truncate = false)
+------------------------------------------------------------------------------------------------+
|col                                                                                             |
+------------------------------------------------------------------------------------------------+
|[chunk, 23, 31, the first, [identifier -> NAME, sentence -> 0, chunk -> 0, entity -> NAME], []] |
|[chunk, 71, 80, ceremonies, [identifier -> NAME, sentence -> 1, chunk -> 0, entity -> NAME], []]|
+------------------------------------------------------------------------------------------------+
{%- endcapture -%}



{%- capture model_api_link -%}
[ChunkConverter](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/chunker/ChunkConverter.html)
{%- endcapture -%}

{%- capture model_python_api_link -%}
[ChunkConverter](https://nlp.johnsnowlabs.com/licensed/api/python/reference/autosummary/sparknlp_jsl/annotator/chunker/chunk_converter/index.html#sparknlp_jsl.annotator.chunker.chunk_converter.ChunkConverter)
{%- endcapture -%}


{% include templates/licensed_approach_model_medical_fin_leg_template.md
title=title
model=model
model_description=model_description
model_input_anno=model_input_anno
model_output_anno=model_output_anno
model_python_medical=model_python_medical
model_scala_medical=model_scala_medical
model_api_link=model_api_link
model_python_api_link=model_python_api_link
%}
