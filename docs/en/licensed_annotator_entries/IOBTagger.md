{%- capture title -%}
IOBTagger
{%- endcapture -%}

{%- capture description -%}
Merges token tags and NER labels from chunks in the specified format.
For example output columns as inputs from
[NerConverter](/docs/en/annotators#nerconverter)
and [Tokenizer](/docs/en/annotators#tokenizer) can be used to merge.
{%- endcapture -%}

{%- capture input_anno -%}
TOKEN, CHUNK
{%- endcapture -%}

{%- capture output_anno -%}
NAMED_ENTITY
{%- endcapture -%}

{%- capture python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.common import *
from sparknlp.annotator import *
from sparknlp.training import *
import sparknlp_jsl
from sparknlp_jsl.base import *
from sparknlp_jsl.annotator import *
from pyspark.ml import Pipeline
# Pipeline stages are defined where NER is done. NER is converted to chunks.
data = spark.createDataFrame([["A 63-year-old man presents to the hospital ..."]]).toDF("text")
docAssembler = DocumentAssembler().setInputCol("text").setOutputCol("document")
sentenceDetector = SentenceDetector().setInputCols(["document"]).setOutputCol("sentence")
tokenizer = Tokenizer().setInputCols(["sentence"]).setOutputCol("token")
embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models").setOutputCol("embs")
nerModel = MedicalNerModel.pretrained("ner_jsl", "en", "clinical/models").setInputCols(["sentence", "token", "embs"]).setOutputCol("ner")
nerConverter = NerConverter().setInputCols(["sentence", "token", "ner"]).setOutputCol("ner_chunk")

# Define the IOB tagger, which needs tokens and chunks as input. Show results.
iobTagger = IOBTagger().setInputCols(["token", "ner_chunk"]).setOutputCol("ner_label")
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

{%- capture scala_example -%}
// Pipeline stages are defined where NER is done. NER is converted to chunks.
val data = Seq(("A 63-year-old man presents to the hospital ...")).toDF("text")
val docAssembler = new DocumentAssembler().setInputCol("text").setOutputCol("document")
val sentenceDetector = new SentenceDetector().setInputCols("document").setOutputCol("sentence")
val tokenizer = new Tokenizer().setInputCols("sentence").setOutputCol("token")
val embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models").setOutputCol("embs")
val nerModel = MedicalNerModel.pretrained("ner_jsl", "en", "clinical/models").setInputCols("sentence", "token", "embs").setOutputCol("ner")
val nerConverter = new NerConverter().setInputCols("sentence", "token", "ner").setOutputCol("ner_chunk")

// Define the IOB tagger, which needs tokens and chunks as input. Show results.
val iobTagger = new IOBTagger().setInputCols("token", "ner_chunk").setOutputCol("ner_label")
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

{%- capture api_link -%}
[IOBTagger](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/ner/IOBTagger)
{%- endcapture -%}

{% include templates/licensed_anno_template.md
title=title
description=description
input_anno=input_anno
output_anno=output_anno
python_example=python_example
scala_example=scala_example
api_link=api_link%}