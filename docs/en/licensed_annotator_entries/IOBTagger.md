{%- capture title -%}
IOBTagger
{%- endcapture -%}

{%- capture model -%}
model
{%- endcapture -%}

{%- capture model_description -%}
Merges token tags and NER labels from chunks in the specified format.
For example output columns as inputs from
[NerConverter](/docs/en/annotators#nerconverter)
and [Tokenizer](/docs/en/annotators#tokenizer) can be used to merge.
{%- endcapture -%}

{%- capture model_input_anno -%}
TOKEN, CHUNK
{%- endcapture -%}

{%- capture model_output_anno -%}
NAMED_ENTITY
{%- endcapture -%}

{%- capture model_python_medical -%}
from johnsnowlabs import * 
# Pipeline stages are defined where NER is done. NER is converted to chunks.
data = spark.createDataFrame([["A 63-year-old man presents to the hospital ..."]]).toDF("text")
docAssembler = nlp.DocumentAssembler().setInputCol("text").setOutputCol("document")
sentenceDetector = nlp.SentenceDetector().setInputCols(["document"]).setOutputCol("sentence")
tokenizer = nlp.Tokenizer().setInputCols(["sentence"]).setOutputCol("token")
embeddings = nlp.WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models").setInputCols(["sentence", "token"]).setOutputCol("embs")
nerModel = medical.NerModel.pretrained("ner_jsl", "en", "clinical/models").setInputCols(["sentence", "token", "embs"]).setOutputCol("ner")
nerConverter = nlp.NerConverter().setInputCols(["sentence", "token", "ner"]).setOutputCol("ner_chunk")

# Define the IOB tagger, which needs tokens and chunks as input. Show results.
iobTagger = medical.IOBTagger().setInputCols(["token", "ner_chunk"]).setOutputCol("ner_label")
pipeline = Pipeline(stages=[docAssembler, sentenceDetector, tokenizer, embeddings, nerModel, nerConverter, iobTagger])

result.selectExpr("explode(ner_label) as a") \
  .selectExpr("a.begin","a.end","a.result as chunk","a.metadata.word as word") \
  .where("chunk!='O'").show(5, False)

+-----+---+-----------+-----------+
|begin|end|chunk      |word       |
+-----+---+-----------+-----------+
|5    |15 |B-Age      |63-year-old|
|17   |19 |B-Gender   |man        |
|64   |72 |B-Modifier |recurrent  |
|98   |107|B-Diagnosis|cellulitis |
|110  |119|B-Diagnosis|pneumonias |
+-----+---+-----------+-----------+

{%- endcapture -%}

{%- capture model_scala_medical -%}
from johnsnowlabs import * 
// Pipeline stages are defined where NER is done. NER is converted to chunks.
val data = Seq(("A 63-year-old man presents to the hospital ...")).toDF("text")
val docAssembler = new nlp.DocumentAssembler().setInputCol("text").setOutputCol("document")
val sentenceDetector = new nlp.SentenceDetector().setInputCols("document").setOutputCol("sentence")
val tokenizer = new nlp.Tokenizer().setInputCols("sentence").setOutputCol("token")
val embeddings = nlp.WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models").setInputCols(Array("sentence", "token")).setOutputCol("embs")
val nerModel = medical.NerModel.pretrained("ner_jsl", "en", "clinical/models").setInputCols(Array("sentence", "token", "embs")).setOutputCol("ner")
val nerConverter = new nlp.NerConverter().setInputCols(Array("sentence", "token", "ner")).setOutputCol("ner_chunk")

// Define the IOB tagger, which needs tokens and chunks as input. Show results.
val iobTagger = new medical.IOBTagger().setInputCols(Array("token", "ner_chunk")).setOutputCol("ner_label")
val pipeline = new Pipeline().setStages(Array(docAssembler, sentenceDetector, tokenizer, embeddings, nerModel, nerConverter, iobTagger))

result.selectExpr("explode(ner_label) as a")
  .selectExpr("a.begin","a.end","a.result as chunk","a.metadata.word as word")
  .where("chunk!='O'").show(5, false)

+-----+---+-----------+-----------+
|begin|end|chunk      |word       |
+-----+---+-----------+-----------+
|5    |15 |B-Age      |63-year-old|
|17   |19 |B-Gender   |man        |
|64   |72 |B-Modifier |recurrent  |
|98   |107|B-Diagnosis|cellulitis |
|110  |119|B-Diagnosis|pneumonias |
+-----+---+-----------+-----------+

{%- endcapture -%}

{%- capture model_api_link -%}
[IOBTagger](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/ner/IOBTagger)
{%- endcapture -%}


{% include templates/licensed_approach_model_medical_fin_leg_template.md
title=title
model=model
model_description=model_description
model_input_anno=model_input_anno
model_output_anno=model_output_anno
model_python_medical=model_python_medical
model_scala_medical=model_scala_medical
model_api_link=model_api_link%}


