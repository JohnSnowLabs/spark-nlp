---
layout: docs
comment: no
header: true
seotitle: Annotation Lab | John Snow Labs
title: Text Annotation
permalink: /docs/en/alab/annotation
key: docs-training
modify_date: "2020-11-18"
use_language_switcher: "Python-Scala"
show_nav: true
sidebar:
    nav: annotation-lab
---

## Overview

The Annotator Lab is designed to keep a human expert as productive as possible. It minimizes the number of mouse clicks, keystrokes, and eye movements in the main workflow, based on iterative feedback from daily users.

Keyboard shortcuts are supported for all annotations – this enables having one hand on keyboard, one hand on mouse, and eyes on screen at all time. One-click completion and automated switching to the next task keeps experts in the loop.

<img class="image image--xl" src="/assets/images/annotation_lab/annotation_main.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

On the upper side of the **Labeling** screen, you can find the list of labels defined for the project. In the center of the screen the content of the task is displayed. On the right side there are several widgets:
- **Completions**
- **Predictions**
- **Confidence**
- **Results**

### Completions

A **completion** is a list of annotations manually defined by a user for a given task. When the work on a task is done (e.g. all entities have been highlighted in the document or the task has been assigned one or more classes in the case of classification projects) the user clicks on the **Save** button. 

Starting Annotation Lab 1.2.0, we introduced the idea of completion submission. In the past, annotators could change or delete completions as many times as they wanted with no restriction. From now on, a submitted completion is no longer editable and cannot be deleted. Creating a new copy of the submitted completion is the only option to edit it. An annotator can modify or delete his/her completions only if the completions are not submitted yet. 

<img class="image image--xl" src="/assets/images/annotation_lab/submit.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

This is an important feature for ensuring a complete audit trail of all user actions. Now, it is possible to track the history and details of any deleted completions, which was not possible in previous releases. This means it is possible to see the name of the completion creator, date of creation, and deletion.

<img class="image image--xl" src="/assets/images/annotation_lab/history.png" style="width:60%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

### Predictions
A **prediction** is a list of annotations created automatically, via the use of Spark NLP pretrained models. Predictions are created using the "Preannotate" button form the **Task** view. Predictions are read only - users can see them but cannot modify them in any way. 
For reusing predictions to bootstrap the annotation process, users can copy them into a new completion which is editable. 

### Confidence
With v3.3.0, running preannotations on a text project provides one extra piece of information for the automatic annotations - the confidence score. This score is used to show the confidence the model has for each of the labeled chunks. It is calculated based on the benchmarking information of the model used to preannotate and on the score of each prediction. The confidence score is available when working on Named Entity Recognition, Relation, Assertion, and Classification projects and is also generated when using NER Rules.

<img class="image image--xl" src="/assets/images/annotation_lab/confidence.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

On the Labeling screen, when selecting the Prediction widget, users can see that all preannotation in the Results section with a score assigned to them . By using the Confidence slider, users can filter out low confidence labels before starting to edit/correct the labels. Both the “Accept Prediction” action and the “Copy Prediction to Completion” feature apply to the filtered annotations via the confidence slider.


### Results
The results widget has two sections. 

The first section - **Regions** - gives a list overview of all annotated chunks. When you click on one annotation it gets automatically highlighted in the document. Annotations can be edited or removed if necessary.

The second section - **Relations** - lists all the relations that have been labeled. When the user moves the mouse over one relation it is highlighted in the document. 

## View As Feature

For users that have multiple roles (Annotator and Reviewer), the labeling view can get confusing. To eliminate this confusion, From 2.6.0, the *View As* filter is added in labeling page too. When selecting *View As Annotator* option, the task is shown as if the only role the currently logged-in user has is Annotator. The same applies to *View As Reviewer*. Once the “View as” option is used to select a certain role, the selection is preserved even when the tab is closed or refreshed. This option is available only if the currently logged-in user has multiple roles. 

<img class="image image--xl" src="/assets/images/annotation_lab/2.6.0/view_as_labeling.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>


## NER Labels
To extract information using NER labels, first click on the label to select it or press the shortcut key assigned to it and then, with the mouse, select the relevant part of the text. Wrong extractions can be easily edited by clicking on them to select them and then by selecting the new label you want to assign to the text. 

<img class="image image--xl" src="/assets/images/annotation_lab/add_label.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

For deleting a label, select it by clicking on it and press backspace. 

### Trim leading and ending spaces in annotated chunks

When annotating text, it is possible and probable that the annotation is not very precise and the chunks contain leading/trailing spaces and punctuation marks. As of 2.8.0 all the leading/trailing spaces and punctuation marks are excluded by default. A new configuration option is added to the Settings in the Labeling page for this purpose. It can be used to disable this feature if necessary.

 ![trim_spaces_punctuations](https://user-images.githubusercontent.com/26042994/158371039-0e5d0722-192e-4d46-81a1-efddba79ccf5.gif)



## Assertion Labels

To add an assertion label to an extracted entity, select the label and use it to do the same exact extraction as the NER label. After this, the extracted entity will have two labels: one for NER and one for assertion. In the example below, the chunks "heart disease", "kidney disease", "stroke" etc. ware extracted using first a blue yellow label and then a red assertion label (**Absent**).

 
<img class="image image--xl" src="/assets/images/annotation_lab/assertion.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>


## Relation Extraction
Creating relations with the Annotation Lab is very simple. First select one labeled entity, then press **r** and select the second labeled entity. 

<img class="image image--xl" src="/assets/images/annotation_lab/relations.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

You can add a label to the relation, change its direction or delete it using the contextual menu displayed next to the relation arrow or using the relation box.
<img class="image image--xl" src="/assets/images/annotation_lab/2.3.0/relations.gif" style="width:50%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>


<img class="image image--xl" src="/assets/images/annotation_lab/relations2.png" style="width:50%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>


## Cross page Annotation

Since release 2.8.0 Annotation Lab supports cross page NER annotation for Text projects. This means that Annotators can annotate a chunk starting at the bottom of one page and finishing on the next page. This feature is also offered for Relations. Previously, relations were created between chunks located on the same page. But now, relations can be created among tokens located on different pages.
The way to do this is to first [change the pagination settings](/docs/en/alab/import#dynamic-task-pagination) to include a higher number of tokens on one page, then create the annotation using the regular annotation approach and finally go back to the original pagination settings. The annotation is persisted after the update of the pagination.
  
 <img class="image image--xl" src="/assets/images/annotation_lab/2.8.0/158549847-b6434489-102e-4245-b751-3f07e6e7c39e.gif" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>


## Pre-annotations  
When you setup a project to use existing Spark NLP models for pre-annotation, you can run the designated models on all of your tasks by pressing the Preannotate button located on the upper side of the task view. 
 
<img class="image image--xl" src="/assets/images/annotation_lab/image028.jpg" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

As a result, all predicted labels for a given task will be available in the Prediction widget, on the main annotation screen. The predictions are not editable, you can only view them and navigate them or compare them with older predictions. However, you can create a new completion based on a given prediction. All labels and relations from such a new completion are now editable. 
 
<img class="image image--xl" src="/assets/images/annotation_lab/image029.png" style="width:50%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>


## Simplified workflow

**Direct Submit:**

Using the classical annotation workflow, when an annotator works on a task, a series of actions are necessary for creating a new annotation and submitting it as ground truth:
-   Create the completion
-   Save the completion, 
-   Submit the completion,
-   Confirm submission, 
-   Load next task. 

This process is adapted for more complex workflows and larger tasks. For cases of simpler projects, involving smaller tasks, Annotation Lab now offers a simplified workflow. Annotators can now submit a completion with just one click.


- The project owner/manager can activate this option from the Settings of the Setup page (Project Configuration). Once enabled, annotators can see the submit button on the labeling page. 
- A second option is available on the same Project Configuration screen for project owner/manager: "Serve next task after completion submission". Once enabled, annotators can see the next task on the labeling page after submitting the completion for the current task.

<img class="image image--xl" src="/assets/images/annotation_lab/2.8.0/158314384-801a8ed5-f0e1-410b-a13e-fc46884ab503.gif" style="width:50%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

--- 

**Accept Prediction:**

When predictions are available for a task, Annotator will be offered the option to accept the predictions with just one click and navigate automatically to the next task. When users click on Accept Prediction, a new completion is created based on the prediction, then submitted as ground truth and the next task in line (assigned to the current annotator/reviewer and with Incomplete or In progress status) is automatically served.

<img class="image image--xl" src="/assets/images/annotation_lab/2.8.0/158314467-bf6c359c-04a7-44d6-8d97-eaeb5314b914.gif" style="width:50%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

---

**Minor update on Shortcuts:**

- Annotator can **save/update** completion using `CTRL+Enter`
- Annotator can **Submit** completion using `ALT+Enter`

---

**Overall work progress:**

Annotator/Reviewer can now see their overall work progress from within the labeling page. The status is calculated with respect to their assigned work. 

- **For Annotator View:**
![image](https://user-images.githubusercontent.com/45035063/158314725-67a5f9fb-8540-4528-8ae3-8cb1ddb7464d.png)

- **For Reviewer View:**
![image](https://user-images.githubusercontent.com/45035063/158314820-b8711b88-b041-4517-a7bc-83a203b6235f.png)




## Annotation Screen customizations

The `Annotation Page` is highly customizable. Project Owners and Managers can change the layout of their projects based on their needs. 

### Search filter for a large number of labels


When a project has a large number of NER/Assertion labels in the taxonomy, the display of the taxonomy takes a lot of screen space and it is difficult for annotators to navigate through all labels. To tackle this challenge, Annotation Lab supports search for labels in NER projects (an autocomplete search option).
To add the search bar for a large number of NER Labels or Choices use the `Filter` tag as shown in the following XML configuration.

```
<Filter/>
<View>

*** enclose labels tags here ***

</View>

**** enclose text tags here**
```

**Parameters:**

The following parameters/attributes can be used within the `Filter` tag.


Param|  Type|   Default Description|
---|----|---|
placeholder|    string  "&quot;Quick Filter&quot;"  |Placeholder text for filter
minlength   |number 3   |Size of the filter
style   |string     |CSS style of the string
hotkey  |string     |Hotkey to use to focus on the filter text area

**Usage Example:**

```
<Filter placeholder="Quick Filter"/>
```



<img class="image image--xl" src="/assets/images/annotation_lab/2.6.0/filter.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

For obtaining the above display on a NER project, the config should look as follows:

```
<View>
    <Filter name="fl" toName="label" hotkey="shift+f" minlength="1" />  
      <Labels name="label" toName="text">
        <Label value="CARDINAL" model="ner_onto_100" background="#af906b"/>
        <Label value="EVENT" model="ner_onto_100" background="#f384e1"/>
        ...
        <Label value="LANGUAGE" model="ner_onto_100" background="#c0dad2"/>
      </Labels>

  <Text name="text" value="$text"/>
</View>
```

Notice how users can search for the desired label using the filter bar:
<img class="image image--xl" src="/assets/images/annotation_lab/2.6.0/ner_label_search.gif" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>


### Resizable labels area and textbox area 

While annotating longer text documents annotators may need to scroll to the top of the document for selecting the label to use, and then  scroll down to create a label. Also, if the text is large, annotators have to scroll to a certain section because the textbox size is fixed. In those cases, the annotation experience can be improved by creating a scrollable labeling area and textbox area.

To add the scroll bar, the `View` tag with `overflow-y:scroll` attribute can be used as shown in the following XML config structure:

```
<View style="background:white; height: 100px; overflow-y:scroll; resize:vertical; position:sticky; top:0;">

*** enclose labels tags here ***

</View>

<View style="resize:vertical; margin-top:10px; max-height:400px; overflow-y:scroll;">

**** enclose text tags here**
</View>
```

Once it has been added and saved to the Project Configuration, the scroll bar should be visible.

<img class="image image--xl" src="/assets/images/annotation_lab/2.6.0/scroll_bar.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

**Example**
Using the following Project Configuration 

```
<View>
  <Filter name="fl" toName="label" hotkey="shift+f" minlength="1" />
  <View style="background:white; height: 100px; overflow-y:scroll; resize:vertical; position:sticky; top:0;">
      <Labels name="label" toName="text">
          <Label value="CARDINAL" model="ner_onto_100" background="#af906b"/>
          <Label value="EVENT" model="ner_onto_100" background="#f384e1"/>
          <Label value="WORK_OF_ART" model="ner_onto_100" background="#0fbca4"/>
          ...
          <Label value="LANGUAGE" model="ner_onto_100" background="#c0dad2"/>
      </Labels>
  </View>
  <View style="resize:vertical; margin-top:10px; max-height:400px; overflow-y:scroll;">
      <Text name="text" value="$text"></Text>
  </View>
</View>
```

we'll obtain the output illustrated below:
<img class="image image--xl" src="/assets/images/annotation_lab/2.6.0/scroll_and_filter.gif" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>
