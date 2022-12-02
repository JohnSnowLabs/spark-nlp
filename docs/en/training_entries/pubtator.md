{%- capture title -%}
PubTator Dataset
{%- endcapture -%}

{%- capture description -%}
The PubTator format includes medical papers' titles, abstracts, and tagged chunks (see [PubTator Docs](http://bioportal.bioontology.org/ontologies/EDAM?p=classes&conceptid=format_3783) and [MedMentions Docs](http://github.com/chanzuckerberg/MedMentions) for more information). We can create a Spark DataFrame from a PubTator text file.
{%- endcapture -%}

{%- capture file_format -%}
25763772        0       5       DCTN4   T116,T123       C4308010
25763772        23      63      chronic Pseudomonas aeruginosa infection        T047    C0854135
25763772        67      82      cystic fibrosis T047    C0010674
25763772        83      120     Pseudomonas aeruginosa (Pa) infection   T047    C0854135
25763772        124     139     cystic fibrosis T047    C0010674
{%- endcapture -%}

{%- capture constructor -%}
None
{%- endcapture -%}

{%- capture read_dataset_params -%}
- **spark**: Initiated Spark Session with Spark NLP
- **path**: Path to the resource
- **isPaddedToken**: Whether tokens are padded
{%- endcapture -%}

{%- capture python_example -%}
from sparknlp.training import PubTator
pubTatorFile = "./src/test/resources/corpus_pubtator_sample.txt"
pubTatorDataSet = PubTator().readDataset(spark, pubTatorFile)
pubTatorDataSet.show(1)
+--------+--------------------+--------------------+--------------------+-----------------------+---------------------+-----------------------+
|  doc_id|      finished_token|        finished_pos|        finished_ner|finished_token_metadata|finished_pos_metadata|finished_label_metadata|
+--------+--------------------+--------------------+--------------------+-----------------------+---------------------+-----------------------+
|25763772|[DCTN4, as, a, mo...|[NNP, IN, DT, NN,...|[B-T116, O, O, O,...|   [[sentence, 0], [...| [[word, DCTN4], [...|   [[word, DCTN4], [...|
+--------+--------------------+--------------------+--------------------+-----------------------+---------------------+-----------------------+
{%- endcapture -%}

{%- capture scala_example -%}
import com.johnsnowlabs.nlp.training.PubTator

val pubTatorFile = "./src/test/resources/corpus_pubtator_sample.txt"
val pubTatorDataSet = PubTator().readDataset(ResourceHelper.spark, pubTatorFile)
pubTatorDataSet.show(1)
+--------+--------------------+--------------------+--------------------+-----------------------+---------------------+-----------------------+
|  doc_id|      finished_token|        finished_pos|        finished_ner|finished_token_metadata|finished_pos_metadata|finished_label_metadata|
+--------+--------------------+--------------------+--------------------+-----------------------+---------------------+-----------------------+
|25763772|[DCTN4, as, a, mo...|[NNP, IN, DT, NN,...|[B-T116, O, O, O,...|   [[sentence, 0], [...| [[word, DCTN4], [...|   [[word, DCTN4], [...|
+--------+--------------------+--------------------+--------------------+-----------------------+---------------------+-----------------------+
{%- endcapture -%}

{%- capture api_link -%}
[PubTator](/api/com/johnsnowlabs/nlp/training/PubTator.html)
{%- endcapture -%}

{%- capture python_api_link -%}
[PubTator](/api/python/reference/autosummary/python/sparknlp/training/pub_tator/index.html#sparknlp.training.pub_tator.PubTator)
{%- endcapture -%}

{%- capture source_link -%}
[PubTator.scala](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/training/PubTator.scala)
{%- endcapture -%}

{% include templates/training_dataset_entry.md
title=title
description=description
file_format=file_format
constructor=constructor
read_dataset_params=read_dataset_params
python_example=python_example
scala_example=scala_example
python_api_link=python_api_link
api_link=api_link
source_link=source_link
%}