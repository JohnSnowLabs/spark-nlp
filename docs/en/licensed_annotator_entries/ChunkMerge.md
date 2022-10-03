{%- capture title -%}
ChunkMerge
{%- endcapture -%}

{%- capture approach -%}
approach
{%- endcapture -%}

{%- capture model -%}
model
{%- endcapture -%}

{%- capture model_description -%}
Merges entities coming from different CHUNK annotations
{%- endcapture -%}

{%- capture model_input_anno -%}
CHUNK, CHUNK
{%- endcapture -%}

{%- capture model_output_anno -%}
CHUNK
{%- endcapture -%}

{%- capture model_api_link -%}
[ChunkMergeModel](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/merge/ChunkMergeModel)
{%- endcapture -%}

{%- capture approach_description -%}
Merges two chunk columns coming from two annotators(NER, ContextualParser or any other annotator producing
chunks). The merger of the two chunk columns is made by selecting one chunk from one of the columns according
to certain criteria.
The decision on which chunk to select is made according to the chunk indices in the source document.
(chunks with longer lengths and highest information will be kept from each source)
Labels can be changed by setReplaceDictResource.
{%- endcapture -%}

{%- capture approach_input_anno -%}
CHUNK, CHUNK
{%- endcapture -%}

{%- capture approach_output_anno -%}
CHUNK
{%- endcapture -%}

{%- capture approach_python_medical -%}
from johnsnowlabs import *
# Define a pipeline with 2 different NER models with a ChunkMergeApproach at the end
data = spark.createDataFrame([["A 63-year-old man presents to the hospital ..."]]).toDF("text")
pipeline = Pipeline(stages=[
 nlp.DocumentAssembler().setInputCol("text").setOutputCol("document"),
 nlp.SentenceDetector().setInputCols(["document"]).setOutputCol("sentence"),
 nlp.Tokenizer().setInputCols(["sentence"]).setOutputCol("token"),
  nlp.WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models").setOutputCol("embs"),
  medical.NerModel.pretrained("ner_jsl", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embs"]).setOutputCol("jsl_ner"),
 nlp.NerConverter().setInputCols(["sentence", "token", "jsl_ner"]).setOutputCol("jsl_ner_chunk"),
  medical.NerModel.pretrained("ner_bionlp", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embs"]).setOutputCol("bionlp_ner"),
 nlp.NerConverter().setInputCols(["sentence", "token", "bionlp_ner"]) \
    .setOutputCol("bionlp_ner_chunk"),
 medical.ChunkMergeApproach().setInputCols(["jsl_ner_chunk", "bionlp_ner_chunk"]).setOutputCol("merged_chunk")
])

# Show results
result = pipeline.fit(data).transform(data).cache()
result.selectExpr("explode(merged_chunk) as a") \
  .selectExpr("a.begin","a.end","a.result as chunk","a.metadata.entity as entity") \
  .show(5, False)
+-----+---+-----------+---------+
|begin|end|chunk      |entity   |
+-----+---+-----------+---------+
|5    |15 |63-year-old|Age      |
|17   |19 |man        |Gender   |
|64   |72 |recurrent  |Modifier |
|98   |107|cellulitis |Diagnosis|
|110  |119|pneumonias |Diagnosis|
+-----+---+-----------+---------+
{%- endcapture -%}

{%- capture approach_python_finance -%}
from johnsnowlabs import *

data = spark.createDataFrame([["Jeffrey Preston Bezos is an American entrepreneur, founder and CEO of Amazon"]]).toDF("text")

documentAssembler = nlp.DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentenceDetector = nlp.SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = nlp.Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

embeddings = nlp.RoBertaEmbeddings.pretrained("roberta_embeddings_legal_roberta_base","en") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("embeddings")

bert_embeddings = nlp.BertEmbeddings.pretrained("bert_embeddings_sec_bert_base","en") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("bert_embeddings")

fin_ner = finance.NerModel.pretrained('finner_deid', "en", "finance/models")\
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner") 
    #.setLabelCasing("upper")

ner_converter =  finance.NerConverterInternal() \
    .setInputCols(["sentence", "token", "ner"])\
    .setOutputCol("ner_chunk")\
    .setReplaceLabels({"ORG": "PARTY"}) # Replace "ORG" entity as "PARTY"

ner_finner = finance.NerModel.pretrained("finner_org_per_role_date", "en", "finance/models")\
    .setInputCols(["sentence", "token", "bert_embeddings"]) \
    .setOutputCol("ner_finner") 
    #.setLabelCasing("upper")

ner_converter_finner = nlp.NerConverter() \
    .setInputCols(["sentence", "token", "ner_finner"]) \
    .setOutputCol("ner_finner_chunk") \
    .setWhiteList(['ROLE']) # Just use "ROLE" entity from this NER

chunk_merge =  finance.ChunkMergeApproach()\
    .setInputCols("ner_finner_chunk", "ner_chunk")\
    .setOutputCol("deid_merged_chunk")

nlpPipeline = Pipeline(stages=[
      documentAssembler, 
      sentenceDetector,
      tokenizer,
      embeddings,
      bert_embeddings,
      fin_ner,
      ner_converter,
      ner_finner,
      ner_converter_finner,
      chunk_merge])

# Show results
result = nlpPipeline.fit(data).transform(data).cache()
result.select(F.explode(F.arrays_zip(result.deid_merged_chunk.result, 
                                     result.deid_merged_chunk.metadata)).alias("cols")) \
      .select(F.expr("cols['0']").alias("chunk"),
              F.expr("cols['1']['entity']").alias("ner_label")).show(truncate=False)
+---------------------+---------+
|chunk                |ner_label|
+---------------------+---------+
|Jeffrey Preston Bezos|PERSON   |
|founder              |ROLE     |
|CEO                  |ROLE     |
|Amazon               |PARTY    |
+---------------------+---------+
{%- endcapture -%}

{%- capture approach_python_legal -%}
from johnsnowlabs import *

data = spark.createDataFrame([["ENTIRE AGREEMENT.  This Agreement contains the entire understanding of the parties hereto with respect to the transactions and matters contemplated hereby, supersedes all previous Agreements between i-Escrow and 2TheMart concerning the subject matter.

2THEMART.COM, INC.:                         I-ESCROW, INC.:

By:Dominic J. Magliarditi                By:Sanjay Bajaj Name: Dominic J. Magliarditi                Name: Sanjay Bajaj Title: President                            Title: VP Business Development Date: 6/21/99                               Date: 6/11/99 "]]).toDF("text")

documentAssembler = nlp.DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentenceDetector = nlp.SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = nlp.Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

embeddings = legal.RoBertaEmbeddings.pretrained("roberta_embeddings_legal_roberta_base","en") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("embeddings")

legal_ner = legal.NerModel.pretrained("legner_contract_doc_parties", "en", "legal/models")\
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner") 
    #.setLabelCasing("upper")

ner_converter = legal.NerConverterInternal() \
    .setInputCols(["sentence", "token", "ner"])\
    .setOutputCol("ner_chunk")\
    .setReplaceLabels({"ALIAS": "PARTY"})

ner_signers = legal.NerModel.pretrained("legner_signers", "en", "legal/models")\
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner_signers") 
    #.setLabelCasing("upper")

ner_converter_signers = nlp.NerConverter() \
    .setInputCols(["sentence", "token", "ner_signers"]) \
    .setOutputCol("ner_signer_chunk")

chunk_merge = legal.ChunkMergeApproach()\
    .setInputCols("ner_signer_chunk", "ner_chunk")\
    .setOutputCol("deid_merged_chunk")

nlpPipeline = Pipeline(stages=[
      documentAssembler, 
      sentenceDetector,
      tokenizer,
      embeddings,
      legal_ner,
      ner_converter,
      ner_signers,
      ner_converter_signers,
      chunk_merge])

# Show results
result = nlpPipeline.fit(data).transform(data).cache()
result.select(F.explode(F.arrays_zip(result.deid_merged_chunk.result, 
                                     result.deid_merged_chunk.metadata)).alias("cols")) \
      .select(F.expr("cols['0']").alias("chunk"),
              F.expr("cols['1']['entity']").alias("ner_label")).show(truncate=False)
+-----------------------+--------------+
|chunk                  |ner_label     |
+-----------------------+--------------+
|ENTIRE AGREEMENT       |DOC           |
|INC                    |PARTY         |
|J. Magliarditi         |SIGNING_PERSON|
|Bajaj                  |SIGNING_PERSON|
|Dominic J. Magliarditi |SIGNING_PERSON|
|Sanjay Bajaj           |SIGNING_PERSON|
|President              |SIGNING_TITLE |
|VP Business Development|SIGNING_TITLE |
+-----------------------+--------------+
{%- endcapture -%}

{%- capture approach_scala_medical -%}
from johnsnowlabs import *

// Define a pipeline with 2 different NER models with a ChunkMergeApproach at the end
val data = Seq(("A 63-year-old man presents to the hospital ...")).toDF("text")
val pipeline = new Pipeline().setStages(Array(
  new nlp.DocumentAssembler().setInputCol("text").setOutputCol("document"),
  new nlp.SentenceDetector().setInputCol("document").setOutputCol("sentence"),
  new nlp.Tokenizer().setInputCol("sentence").setOutputCol("token"),
  nlp.WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models").setInputCols(Array("sentence","token")).setOutputCol("embs"),
  medical.NerModel.pretrained("ner_jsl", "en", "clinical/models")
    .setInputCols(Array("sentence", "token", "embs")).setOutputCol("jsl_ner"),
  new nlp.NerConverter().setInputCols(Array("sentence", "token", "jsl_ner")).setOutputCol("jsl_ner_chunk"),
  medical.NerModel.pretrained("ner_bionlp", "en", "clinical/models")
    .setInputCols(Array("sentence", "token", "embs")).setOutputCol("bionlp_ner"),
  new nlp.NerConverter().setInputCols(Array("sentence", "token", "bionlp_ner"))
    .setOutputCol("bionlp_ner_chunk"),
  new medical.ChunkMergeApproach().setInputCols(Array("jsl_ner_chunk", "bionlp_ner_chunk")).setOutputCol("merged_chunk")
))

// Show results
val result = pipeline.fit(data).transform(data).cache()
result.selectExpr("explode(merged_chunk) as a")
  .selectExpr("a.begin","a.end","a.result as chunk","a.metadata.entity as entity")
  .show(5, false)
+-----+---+-----------+---------+
|begin|end|chunk      |entity   |
+-----+---+-----------+---------+
|5    |15 |63-year-old|Age      |
|17   |19 |man        |Gender   |
|64   |72 |recurrent  |Modifier |
|98   |107|cellulitis |Diagnosis|
|110  |119|pneumonias |Diagnosis|
+-----+---+-----------+---------+
{%- endcapture -%}

{%- capture approach_scala_finance -%}
from johnsnowlabs import *

val data = Seq(("Jeffrey Preston Bezos is an American entrepreneur, founder and CEO of Amazon")).toDF("text")

val documentAssembler = new nlp.DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val sentenceDetector = new nlp.SentenceDetector()
    .setInputCol("document")
    .setOutputCol("sentence")

val tokenizer = new nlp.Tokenizer()
    .setInputCol("sentence")
    .setOutputCol("token")

val embeddings = nlp.RoBertaEmbeddings.pretrained("roberta_embeddings_legal_roberta_base","en")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("embeddings")

val bert_embeddings = nlp.BertEmbeddings.pretrained("bert_embeddings_sec_bert_base","en")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("bert_embeddings")

val fin_ner = finance.NerModel.pretrained('finner_deid', "en", "finance/models")
    .setInputCols(Array("sentence", "token", "embeddings"))
    .setOutputCol("ner") 
    #.setLabelCasing("upper")

val ner_converter =  finance.NerConverterInternal()
    .setInputCols(Array("sentence", "token", "ner"))
    .setOutputCol("ner_chunk")\
    .setReplaceLabels({"ORG": "PARTY"}) # Replace "ORG" entity as "PARTY"

val ner_finner = finance.NerModel.pretrained("finner_org_per_role_date", "en", "finance/models")\
    .setInputCols(Array("sentence", "token", "bert_embeddings"))
    .setOutputCol("ner_finner") 
    #.setLabelCasing("upper")

val ner_converter_finner = new nlp.NerConverter()
    .setInputCols(Array("sentence", "token", "ner_finner"))
    .setOutputCol("ner_finner_chunk")
    .setWhiteList(['ROLE']) # Just use "ROLE" entity from this NER

val chunk_merge =  new finance.ChunkMergeApproach()
    .setInputCols(Array("ner_finner_chunk", "ner_chunk"))
    .setOutputCol("deid_merged_chunk")

val nlpPipeline = new Pipeline().setStages(Array(
      documentAssembler, 
      sentenceDetector,
      tokenizer,
      embeddings,
      bert_embeddings,
      fin_ner,
      ner_converter,
      ner_finner,
      ner_converter_finner,
      chunk_merge))

val model = nlpPipeline.fit(data)
{%- endcapture -%}

{%- capture approach_scala_legal -%}
from johnsnowlabs import *

val data = Seq(("ENTIRE AGREEMENT.  This Agreement contains the entire understanding of the parties hereto with respect to the transactions and matters contemplated hereby, supersedes all previous Agreements between i-Escrow and 2TheMart concerning the subject matter.

2THEMART.COM, INC.:                         I-ESCROW, INC.:

By:Dominic J. Magliarditi                By:Sanjay Bajaj Name: Dominic J. Magliarditi                Name: Sanjay Bajaj Title: President                            Title: VP Business Development Date: 6/21/99                               Date: 6/11/99 ")).toDF("text")

val documentAssembler = new nlp.DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val sentenceDetector = new nlp.SentenceDetector()
    .setInputCol("document")
    .setOutputCol("sentence")

val tokenizer = new nlp.Tokenizer()
    .setInputCol("sentence")
    .setOutputCol("token")

val embeddings = nlp.RoBertaEmbeddings.pretrained("roberta_embeddings_legal_roberta_base","en")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("embeddings")

val legal_ner = legal.NerModel.pretrained("legner_contract_doc_parties", "en", "legal/models")
    .setInputCols(Array("sentence", "token", "embeddings"))
    .setOutputCol("ner") 
    #.setLabelCasing("upper")

val ner_converter = new legal.NerConverterInternal()
    .setInputCols(Array("sentence", "token", "ner"))
    .setOutputCol("ner_chunk")\
    .setReplaceLabels({"ALIAS": "PARTY"})

val ner_signers = legal.NerModel.pretrained("legner_signers", "en", "legal/models")
    .setInputCols(Array("sentence", "token", "embeddings"))
    .setOutputCol("ner_signers") 
    #.setLabelCasing("upper")

val ner_converter_signers = new nlp.NerConverter()
    .setInputCols(Array("sentence", "token", "ner_signers"))
    .setOutputCol("ner_signer_chunk")

val chunk_merge = new legal.ChunkMergeApproach()
    .setInputCols(Array("ner_signer_chunk", "ner_chunk"))
    .setOutputCol("deid_merged_chunk")

val nlpPipeline = new Pipeline().setStages(Array(
      documentAssembler, 
      sentenceDetector,
      tokenizer,
      embeddings,
      legal_ner,
      ner_converter,
      ner_signers,
      ner_converter_signers,
      chunk_merge))

val model = nlpPipeline.fit(data)
{%- endcapture -%}

{%- capture approach_api_link -%}
[ChunkMergeApproach](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/merge/ChunkMergeApproach)
{%- endcapture -%}


{% include templates/licensed_approach_model_medical_fin_leg_template.md
title=title
model=model
approach=approach
model_description=model_description
model_input_anno=model_input_anno
model_output_anno=model_output_anno
model_api_link=model_api_link
approach_description=approach_description
approach_input_anno=approach_input_anno
approach_output_anno=approach_output_anno
approach_python_medical=approach_python_medical
approach_python_finance=approach_python_finance
approach_python_legal=approach_python_legal
approach_scala_medical=approach_scala_medical
approach_scala_finance=approach_scala_finance
approach_scala_legal=approach_scala_legal
approach_api_link=approach_api_link
%}
