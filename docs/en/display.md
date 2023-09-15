---
layout: docs
header: true
seotitle: Spark NLP - Spark NLP Display
title: Spark NLP - Spark NLP Display
permalink: /docs/en/display
key: docs-display
modify_date: "2020-11-17"
show_nav: true
sidebar:
    nav: sparknlp
---

<div class="h3-box" markdown="1">

### Getting started

Spark NLP Display is an open-source python library for visualizing the annotations generated with Spark NLP. It currently offers out-of-the-box suport for the following types of annotations:
- Dependency Parser
- Named Entity Recognition
- Entity Resolution
- Relation Extraction
- Assertion Status

The ability to quickly visualize the entities/relations/assertion statuses, etc. generated using Spark NLP is a very useful feature for speeding up the development process as well as for understanding the obtained results. Getting all of this in a one liner is extremelly convenient especially when running Jupyter notebooks which offers full support for html visualizations.


The  visualisation classes work with the outputs returned by both Pipeline.transform() function and LightPipeline.fullAnnotate().


</div><div class="h3-box" markdown="1">

### Install Spark NLP Display

You can install the Spark NLP Display library via pip by using:

```bash
pip install spark-nlp-display
```

<br/>

A complete guideline on how to use the Spark NLP Display library is available <a href="https://github.com/JohnSnowLabs/spark-nlp-display/blob/main/tutorials/Spark_NLP_Display.ipynb">here</a>.

</div><div class="h3-box" markdown="1">

### Visualize a dependency tree

For visualizing a dependency trees generated with <a href="https://sparknlp.org/docs/en/annotators#dependency-parsers">DependencyParserApproach</a> you can use the following code.


```bash
from sparknlp_display import DependencyParserVisualizer

dependency_vis = DependencyParserVisualizer()

dependency_vis.display(pipeline_result[0], #should be the results of a single example, not the complete dataframe.
                       pos_col = 'pos', #specify the pos column
                       dependency_col = 'dependency', #specify the dependency column
                       dependency_type_col = 'dependency_type' #specify the dependency type column
                       )
```
<br/>

The following image gives an example of html output that is obtained for a test sentence:

<img class="image image--xl" src="/assets/images/dependency tree viz.png" style="width:70%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>


</div><div class="h3-box" markdown="1">

### Visualize extracted named entities

The **NerVisualizer** highlights the named entities that are identified by Spark NLP and also displays their labels as decorations on top of the analyzed text. The colors assigned to the predicted labels can be configured to fit the particular needs of the application.

```bash
from sparknlp_display import NerVisualizer

ner_vis = NerVisualizer()

ner_vis.display(pipeline_result[0], #should be the results of a single example, not the complete dataframe
                    label_col='entities', #specify the entity column
                    document_col='document' #specify the document column (default: 'document')
                    labels=['PER'] #only allow these labels to be displayed. (default: [] - all labels will be displayed)
                    )

## To set custom label colors:
ner_vis.set_label_colors({'LOC':'#800080', 'PER':'#77b5fe'}) #set label colors by specifying hex codes
```
The following image gives an example of html output that is obtained for a couple of test sentences:

<img class="image image--xl" src="/assets/images/ner viz.png" style="width:80%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>


</div><div class="h3-box" markdown="1">

### Visualize relations

The **RelationExtractionVisualizer** can be used to visualize the relations predicted by Spark NLP. The two entities involved in a relation will be highlighted and their label will be displayed. Also a directed and labeled arc(line) will be used to connect the two entities.


```bash
from sparknlp_display import RelationExtractionVisualizer

re_vis = RelationExtractionVisualizer()

re_vis.display(pipeline_result[0], #should be the results of a single example, not the complete dataframe
               relation_col = 'relations', #specify relations column
               document_col = 'document', #specify document column
               show_relations=True #display relation names on arrows (default: True)
               )
```
The following image gives an example of html output that is obtained for a couple of test sentences:

<img class="image image--xl" src="/assets/images/relations viz.png" style="width:100%;align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

</div><div class="h3-box" markdown="1">

### Visualize assertion status

The **AssertionVisualizer** is a special type of **NerVisualizer** that also displays on top of the labeled entities the assertion status that was infered  by a Spark NLP model.


```bash
from sparknlp_display import AssertionVisualizer

assertion_vis = AssertionVisualizer()

assertion_vis.display(pipeline_result[0],
                      label_col = 'entities', #specify the ner result column
                      assertion_col = 'assertion' #specify assertion column
                      document_col = 'document' #specify the document column (default: 'document')
                      )

## To set custom label colors:
assertion_vis.set_label_colors({'TREATMENT':'#008080', 'problem':'#800080'}) #set label colors by specifying hex codes

```
The following image gives an example of html output that is obtained for a couple of test sentences:

<img class="image image--xl" src="/assets/images/assertion viz.png" style="width:80%;align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

</div><div class="h3-box" markdown="1">

### Visualize entity resolution

**Entity resolution** refers to the normalization of named entities predicted by Spark NLP with respect to standard terminologies such as ICD-10, SNOMED, RxNorm etc. You can read more about the available entity resolvers <a href="/en/licensed_annotators#chunkentityresolver">here.</a>

The **EntityResolverVisualizer** will automatically display on top of the NER label the standard code (ICD10 CM, PCS, ICDO; CPT) that corresponds to that entity as well as the short description of the code. If no resolution code could be identified a regular NER-type of visualization will be displayed.

```bash
from sparknlp_display import EntityResolverVisualizer

er_vis = EntityResolverVisualizer()

er_vis.display(pipeline_result[0], #should be the results of a single example, not the complete dataframe
               label_col='entities', #specify the ner result column
               resolution_col = 'resolution'
               document_col='document' #specify the document column (default: 'document')
               )

## To set custom label colors:
er_vis.set_label_colors({'TREATMENT':'#800080', 'PROBLEM':'#77b5fe'}) #set label colors by specifying hex codes
```


The following image gives an example of html output that is obtained for a couple of test sentences:

<img class="image image--xl" src="/assets/images/resolution viz.png" style="width:100%;align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

</div>