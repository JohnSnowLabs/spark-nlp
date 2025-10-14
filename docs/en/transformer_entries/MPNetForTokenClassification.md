{%- capture title -%}
MPNetForTokenClassification
{%- endcapture -%}

{%- capture description -%}
The MPNet model was proposed in [MPNet: Masked and Permuted Pre-training for Language Understanding](https://huggingface.co/papers/2004.09297) by Kaitao Song, Xu Tan, Tao Qin, Jianfeng Lu, Tie-Yan Liu.

MPNet adopts a novel pre-training method, named masked and permuted language modeling, to inherit the advantages of masked language modeling and permuted language modeling for natural language understanding.

`MPNetForTokenClassification` in Spark NLP is a token classification annotator based on the MPNet architecture. It can be used for tasks such as Named Entity Recognition (NER), Part-of-Speech (POS) tagging, and other token-level classification problems.  

Pretrained models can be loaded with the `pretrained` method of the companion object:
```scala
val tokenClassifier = MPNetForTokenClassification.pretrained("mpnet_base_token_classifier", "en")
    .setInputCols(Array("documents","token")) 
    .setOutputCol("ner")
```

For available pretrained models please see the [Models Hub](https://sparknlp.org/models?annotator=MPNetForTokenClassification).

Spark NLP supports a variety of Hugging Face Transformers for token classification. To learn how to import and use them, check out the following thread:
- [Import models into Spark NLP](https://github.com/JohnSnowLabs/spark-nlp/discussions/5669)

**Resources**:

- [MPNet: Masked and Permuted Pre-training for Language Understanding (Paper)](https://arxiv.org/abs/2004.09297)  

**Paper abstract**

*BERT adopts masked language modeling (MLM) for pre-training and is one of the most successful pre-training models. Since BERT neglects dependency among predicted tokens, XLNet introduces permuted language modeling (PLM) for pre-training to address this problem. However, XLNet does not leverage the full position information of a sentence and thus suffers from position discrepancy between pre-training and fine-tuning. In this paper, we propose MPNet, a novel pre-training method that inherits the advantages of BERT and XLNet and avoids their limitations. MPNet leverages the dependency among predicted tokens through permuted language modeling (vs. MLM in BERT), and takes auxiliary position information as input to make the model see a full sentence and thus reducing the position discrepancy (vs. PLM in XLNet). We pre-train MPNet on a large-scale dataset (over 160GB text corpora) and fine-tune on a variety of down-streaming tasks (GLUE, SQuAD, etc). Experimental results show that MPNet outperforms MLM and PLM by a large margin, and achieves better results on these tasks compared with previous state-of-the-art pre-trained methods (e.g., BERT, XLNet, RoBERTa) under the same model setting.*
{%- endcapture -%}

{%- capture input_anno -%}
TOKEN, DOCUMENT
{%- endcapture -%}

{%- capture output_anno -%}
CATEGORY
{%- endcapture -%}

{%- capture api_link -%}
[MPNetForTokenClassification](/api/com/johnsnowlabs/nlp/annotators/classifier/dl/MPNetForTokenClassification.html)
{%- endcapture -%}

{%- capture python_api_link -%}
[MPNetForTokenClassification](api/python/reference/autosummary/sparknlp/annotator/classifier_dl/mpnet_for_token_classification/index.html)
{%- endcapture -%}

{%- capture source_link -%}
[MPNetForTokenClassification](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/MPNetForTokenClassification.scala)
{%- endcapture -%}

{%- capture python_example -%}
from sparknlp.base import DocumentAssembler
from sparknlp.annotator import Tokenizer, MPNetForTokenClassification
from pyspark.ml import Pipeline

document_assembler = DocumentAssembler() \\
    .setInputCol("text") \\
    .setOutputCol("document")

tokenizer = Tokenizer() \\
    .setInputCols(["document"]) \\
    .setOutputCol("token")

tokenClassifier = MPNetForTokenClassification.pretrained("token_classifier_mpnet_base", "en") \\
    .setInputCols(["token", "document"]) \\
    .setOutputCol("ner")

pipeline = Pipeline().setStages([
    document_assembler,
    tokenizer,
    tokenClassifier
])

data = spark.createDataFrame([["John lives in New York."]], ["text"])

model = pipeline.fit(data)
results = model.transform(data)

results.selectExpr("explode(ner.result) as ner").show(truncate=False)
{%- endcapture -%}

{%- capture scala_example -%}
import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.{Tokenizer, MPNetForTokenClassification}
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val tokenizer = new Tokenizer()
  .setInputCols("document")
  .setOutputCol("token")

val tokenClassifier = MPNetForTokenClassification.pretrained("token_classifier_mpnet_base", "en")
  .setInputCols("token", "document")
  .setOutputCol("ner")

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  tokenizer,
  tokenClassifier
))

val data = Seq("John lives in New York.").toDF("text")

val model = pipeline.fit(data)
val results = model.transform(data)

results.selectExpr("explode(ner.result) as ner").show(false)
{%- endcapture -%}

{% include templates/anno_template.md
title=title
description=description
input_anno=input_anno
output_anno=output_anno
python_example=python_example
scala_example=scala_example
api_link=api_link
python_api_link=python_api_link
source_link=source_link
%}