{%- capture title -%}
DistilBertEmbeddings
{%- endcapture -%}

{%- capture description -%}
DistilBERT is a small, fast, cheap and light Transformer model trained by distilling BERT base. It has 40% less parameters than
`bert-base-uncased`, runs 60% faster while preserving over 95% of BERT's performances as measured on the GLUE language understanding benchmark.

Pretrained models can be loaded with `pretrained` of the companion object:
```
val embeddings = DistilBertEmbeddings.pretrained()
  .setInputCols("document", "token")
  .setOutputCol("embeddings")
```
The default model is `"distilbert_base_cased"`, if no name is provided.
For available pretrained models please see the [Models Hub](https://nlp.johnsnowlabs.com/models?task=Embeddings).

For extended examples of usage, see the [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/transformers/HuggingFace%20in%20Spark%20NLP%20-%20DistilBERT.ipynb)
and the [DistilBertEmbeddingsTestSpec](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/test/scala/com/johnsnowlabs/nlp/embeddings/DistilBertEmbeddingsTestSpec.scala).
Models from the HuggingFace 🤗 Transformers library are also compatible with Spark NLP 🚀. The Spark NLP Workshop
example shows how to import them.

The DistilBERT model was proposed in the paper
[DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108).

**Paper Abstract:**

''As Transfer Learning from large-scale pre-trained models becomes more prevalent in Natural Language Processing (NLP),
operating these large models in on-the-edge and/or under constrained computational training or inference budgets
remains challenging. In this work, we propose a method to pre-train a smaller general-purpose language representation
model, called DistilBERT, which can then be fine-tuned with good performances on a wide range of tasks like its larger
counterparts. While most prior work investigated the use of distillation for building task-specific models, we leverage
knowledge distillation during the pretraining phase and show that it is possible to reduce the size of a BERT model by
40%, while retaining 97% of its language understanding capabilities and being 60% faster. To leverage the inductive
biases learned by larger models during pretraining, we introduce a triple loss combining language modeling,
distillation and cosine-distance losses. Our smaller, faster and lighter model is cheaper to pre-train and we
demonstrate its capabilities for on-device computations in a proof-of-concept experiment and a comparative on-device
study.''

Tips:
  - DistilBERT doesn't have `:obj:token_type_ids`, you don't need to indicate which token belongs to which segment. Just
    separate your segments with the separation token `:obj:tokenizer.sep_token` (or `:obj:[SEP]`).
  - DistilBERT doesn't have options to select the input positions (`:obj:position_ids` input). This could be added if
    necessary though, just let us know if you need this option.
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT, TOKEN
{%- endcapture -%}

{%- capture output_anno -%}
WORD_EMBEDDINGS
{%- endcapture -%}

{%- capture python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.common import *
from sparknlp.annotator import *
from sparknlp.training import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")

embeddings = DistilBertEmbeddings.pretrained() \
    .setInputCols(["document", "token"]) \
    .setOutputCol("embeddings") \
    .setCaseSensitive(True)

embeddingsFinisher = EmbeddingsFinisher() \
    .setInputCols(["embeddings"]) \
    .setOutputCols("finished_embeddings") \
    .setOutputAsVector(True) \
    .setCleanAnnotations(False)

pipeline = Pipeline() \
    .setStages([
      documentAssembler,
      tokenizer,
      embeddings,
      embeddingsFinisher
    ])

data = spark.createDataFrame([["This is a sentence."]]).toDF("text")
result = pipeline.fit(data).transform(data)

result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
+--------------------------------------------------------------------------------+
|                                                                          result|
+--------------------------------------------------------------------------------+
|[0.1127224713563919,-0.1982710212469101,0.5360898375511169,-0.272536993026733...|
|[0.35534414649009705,0.13215228915214539,0.40981462597846985,0.14036104083061...|
|[0.328085333108902,-0.06269335001707077,-0.017595693469047546,-0.024373905733...|
|[0.15617232024669647,0.2967822253704071,0.22324979305267334,-0.04568954557180...|
|[0.45411425828933716,0.01173491682857275,0.190129816532135,0.1178255230188369...|
+--------------------------------------------------------------------------------+

{%- endcapture -%}

{%- capture scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.embeddings.DistilBertEmbeddings
import com.johnsnowlabs.nlp.EmbeddingsFinisher
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val tokenizer = new Tokenizer()
  .setInputCols(Array("document"))
  .setOutputCol("token")

val embeddings = DistilBertEmbeddings.pretrained()
  .setInputCols("document", "token")
  .setOutputCol("embeddings")
  .setCaseSensitive(true)

val embeddingsFinisher = new EmbeddingsFinisher()
  .setInputCols("embeddings")
  .setOutputCols("finished_embeddings")
  .setOutputAsVector(true)
  .setCleanAnnotations(false)

val pipeline = new Pipeline()
  .setStages(Array(
    documentAssembler,
    tokenizer,
    embeddings,
    embeddingsFinisher
  ))

val data = Seq("This is a sentence.").toDF("text")
val result = pipeline.fit(data).transform(data)

result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
+--------------------------------------------------------------------------------+
|                                                                          result|
+--------------------------------------------------------------------------------+
|[0.1127224713563919,-0.1982710212469101,0.5360898375511169,-0.272536993026733...|
|[0.35534414649009705,0.13215228915214539,0.40981462597846985,0.14036104083061...|
|[0.328085333108902,-0.06269335001707077,-0.017595693469047546,-0.024373905733...|
|[0.15617232024669647,0.2967822253704071,0.22324979305267334,-0.04568954557180...|
|[0.45411425828933716,0.01173491682857275,0.190129816532135,0.1178255230188369...|
+--------------------------------------------------------------------------------+

{%- endcapture -%}

{%- capture api_link -%}
[DistilBertEmbeddings](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/embeddings/DistilBertEmbeddings)
{%- endcapture -%}

{%- capture source_link -%}
[DistilBertEmbeddings](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/embeddings/DistilBertEmbeddings.scala)
{%- endcapture -%}

{% include templates/anno_template.md
title=title
description=description
input_anno=input_anno
output_anno=output_anno
python_example=python_example
scala_example=scala_example
api_link=api_link
source_link=source_link
%}