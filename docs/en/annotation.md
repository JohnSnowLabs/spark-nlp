---
layout: docs
comment: no
header: true
title: Text Annotation
permalink: /docs/en/annotation
key: docs-training
modify_date: "2020-11-18"
use_language_switcher: "Python-Scala"
---

## Overview

The Annotator Lab is designed to keep a human expert as productive as possible. It minimizes the number of mouse clicks, keystrokes, and eye movements in the main workflow, based on iterative feedback from daily users.

Keyboard shortcuts are supported for all annotations – this enables having one hand on keyboard, one hand on mouse, and eyes on screen at all time. One-click completion and automated switching to the next task keeps experts in the loop.

<img class="image image--xl" src="/assets/images/annotation_lab/annotation_main.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

On the upper side of the **Labeling** screen, you can find the list of labels defined for the project. In the center of the screen the content of the task is displayed. On the right side there are several widgets:
- **Completions**
- **Predictions**
- **Results**

### Completions

A **completion** is a list of annotations manually defined by a user for a given task. When the work on a task is done (e.g. all entities have been highlighted in the document or the task has been assigned one or more classes in the case of classification projects) the user clicks on the **Submit** button. This will create a new completion and will automatically load the next task for annotation. 

### Predictions
A **prediction** is a list of annotations created automatically, via the use of Spark NLP pretrained models. Predictions are created using the "Preannotate" button form the **Task** view. Predictions are read only - users can see them but cannot modify them in any way. 

### Results
The results widget has two sections. 

The first section - **Regions** - gives a list overview of all annotated chunks. When you click on one annotation it gets automatically highlighted in the document. Annotations can be edited or removed if necessary.

The second section - **Relations** - lists all the relations that have been labeled. When the user moves the mouse over one relation it is highlighted in the document. 


## NER Labels
To extract information using NER labels, first click on the label to select it or press the shortcut key assigned to it and then, with the mouse, select the relevant part of the text. Wrong extractions can be easily edited by clicking on them to select them and then by selecting the new label you want to assign to the text. 

<img class="image image--xl" src="/assets/images/annotation_lab/add_label.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

For deleting a label, select it by clicking on it and press backspace. 
 
## Assertion Labels

To add an assertion label to an extracted entity, select the label and use it to do the same exact extraction as the NER label. After this, the extracted entity will have two labels: one for NER and one for assertion. In the example below, the chunks "heart disease", "kidney disease", "stroke" etc. ware extracted using first a blue yellow label and then a red assertion label (**Absent**).

 
<img class="image image--xl" src="/assets/images/annotation_lab/assertion.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>


## Relation Extraction
Creating relations with the Annotation Lab is very simple. First select one labeled entity, then press **r** and select the second labeled entity. 

<img class="image image--xl" src="/assets/images/annotation_lab/relations.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

You can add a label to the relation, change its direction or delete it using the relation box.

<img class="image image--xl" src="/assets/images/annotation_lab/relations2.png" style="width:50%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

 
## Pre-annotations  
When you setup a project to use existing Spark NLP models for pre-annotation, you can run the designated models on all of your tasks by pressing the Preannotate button located on the upper side of the task view. 
 
<img class="image image--xl" src="/assets/images/annotation_lab/image028.jpg" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

As a result, all predicted labels for a given task will be available in the Prediction widget, on the main annotation screen. The predictions are not editable, you can only view them and navigate them or compare them with older predictions. However, you can create a new completion based on a given prediction. All labels and relations from such a new completion are now editable. 
 
<img class="image image--xl" src="/assets/images/annotation_lab/image029.png" style="width:50%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>