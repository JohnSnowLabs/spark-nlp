---
layout: docs
comment: no
header: true
title: Project Setup 
permalink: /docs/en/project_setup
key: docs-training
modify_date: "2020-11-18"
use_language_switcher: "Python-Scala"
---

To create a new project, click on the **Create Project** button on the **Home Page** and choose a name for it. The project can include a short description and annotation instructions/guidelines. 

<img class="image image--xl" src="/assets/images/annotation_lab/1.6.0/project_creation.png" style="width:100%; align:left; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>


## Share your project with the annotation team

When working in teams, projects can be shared with other team members. 

 <img class="image image--xl" src="/assets/images/annotation_lab/1.6.0/project_sharing.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

The user who creates a project is called a **Project Owner**. He/She has complete visibility and ownership of the project for its entire lifecycle. If the Project Owner is removed from the user database, then all his/her projects are transfered to a new project owner. The Project Owner can edit the project configuration, can import/export tasks, can create a project team that will work on his project and can access project analytics.   
When defining the project team, a project owner has access to three distinct roles: **Annotator**, **Reviewer**, and **Manager**. These are very useful for most of the workflows that our users follow. 
 - An **Annotator** is able to see the tasks which have been assigned to him or her and can create annotations on the documents. 
 - The **Reviewer** is able to see the work of the annotators and approve it or reject in case he finds issues that need to be solved.  
 - The **Manager** is able to see the work of the Annotators and of the Reviewers and he can assign tasks to team members. This is useful for eliminating work overlap and for a better management of the work load.

To add a user to your project team, navigate to the Project Setup page. On the Manage Project Team tab, start typing the name of a user in the available text box. This will populate a list of available users having the username start with the caracters you typed. From the dropdown select the user you want to add to your team. Select a role for the user and click on the "Add to team" button. 

 <img class="image image--xl" src="/assets/images/annotation_lab/1.6.0/user_management.gif" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>


## Supported Project Types
We currently support multiple predefined project configurations. The most popular ones are **Text Classification** and **Named Entity Recognition**. Create a setup from scratch or customize a predefined one according to your needs.


 <img class="image image--xl" src="/assets/images/annotation_lab/predefined_configurations.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

 
For customizing a predefined configuration, click on the corresponding link in the table above and then navigate to the Labeling config widget and manually edit/update it to contain the labels you need. 
 
After you finish editing the labels you want to define for your project click the “Save” button. 
 
## Text Classification Project

The Annotation Lab offers two types of classification widgets:

The first one supports single choice labels. You can activate it by choosing **Text Classification** from the list of predefined projects. The labels can be changed by directly editing them in the **Labeling Config** XML style widget. The updates will be automatically reflected in the right side preview. 

<img class="image image--xl" src="/assets/images/annotation_lab/sent_analysis.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>


The second configuration offers support for multi-class classification. It can be activated by clicking on the **Multi classification** link in the list of predefined configurations. This option will add to the labeling config widget multiple checkboxes, grouped by headers. The names of the choices and well as the headers are customizable. You can also add new choices if necessary. 

<img class="image image--xl" src="/assets/images/annotation_lab/image013.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

 
## Named Entity Recognition Project

Named Entity Recognition (NER) refers to the identification and classification of entities mentioned in unstructured text into pre-defined categories such as person names, organizations, locations, medical codes, time expressions, quantities, monetary values, percentages, etc.


The **Annotation Lab** offers support for two types of labels: 
-	Simple labels for NER or assertion models;
-	Binary relations for relation extraction models. 

<img class="image image--xl" src="/assets/images/annotation_lab/labels_def.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

## Assertion Status Project
The syntax for defining an Assertion Status label is the same as for the NER labels, with an additional attribute - **assertion** which should be set to true (see example below). This is convention defined by Annotation Lab users which we exploited for identifying the labels to include in the training and prediction of Assertion Models.
A simple Labeling Config with Assertion Status defined should look like the following:

```bash
<View>
<Labels name="ner" toName="text">
	<Label value="Medicine" background="orange" hotkey="_"/>
	<Label value="Condition" background="orange" hotkey="_"/>
	<Label value="Procedure" background="green" hotkey="8"/>
	<Label value="Absent" assertion="true" background="red" hotkey="Z"/>
	<Label value="Past" assertion="true" background="red" hotkey="X"/>
</Labels>
<View style="height: 250px; overflow: auto;">
	<Text name="text" value="$text"/>
</View>
</View>
```
Notice assertion="true" in **Absent** and **Past** labels, which marks each of those labels as Assertion Status Labels.

## Labels customization
-	Names of the labels must be carefully chosen so they are easy to understand by the annotators. 
-	Highlighting colors can be assigned to each labels by either specifying the color name or the color code. 
-	Shortcuts keys can be assigned to each label to make the annotation process easier and faster. 

```bash
<Labels name="ner" toName="text">
    <Label value="Cancer" background="red" hotkey="c"/>
    <Label value="TumorSize" background="blue" hotkey="t"/>
    <Label value="TumorLocation" background="pink" hotkey="l"/>
    <Label value="Symptom" background="#dda0dd" hotkey="z"/>
  </Labels>
```

## Relations
The Annotation Lab also offers support for relation extraction. Relations are introduced by simply specifying their label. 

```bash
<Relations>
    <Relation value="CancerSize" />
    <Relation value="CancerLocation"/>
    <Relation value="MetastasisLocation"/>
  </Relations>
```
No other constraints can currently be enforced on the labels linked by the defined relations so the annotators must be extra careful and follow the annotation guidelines that specify how the defined relations can be used.  
 