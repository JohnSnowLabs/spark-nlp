---
layout: docs
comment: no
header: true
seotitle: Annotation Lab | John Snow Labs
title: Project
permalink: /docs/en/alab/project
key: docs-training
modify_date: "2022-10-13"
use_language_switcher: "Python-Scala"
show_nav: true
sidebar:
    nav: annotation-lab
---
## Dashboard
When you login to the Annotation Lab, you will be presented with the the project dashboard. For each project details like description, task counts, groups, team members, etc. are available on the main dashboard so users can quickly identify the projects they need to work on, without navigating to the Project Details page. You will be presented with the list of available projects you have created **My Projects** or that have been shared with you **Shared With Me**. **All Projects** combines the list of the projects created by you and the the projects shared to the you.

<img class="image image--xl" src="/assets/images/annotation_lab/4.1.0/dashboardShared.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>


The list of projects can be sorted according to the name of the project. Also, projects can be sorted in ascending or descending order according to the creation date. name. 

<img class="image image--xl" src="/assets/images/annotation_lab/4.1.0/dashboardSort.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

The filters associated with the Projects dashboard are clear, simple, and precise to make the users more productive and efficient while working with a large number of projects.

![DashboardFilterGIF](https://user-images.githubusercontent.com/46840490/193030380-df251a49-07fa-48a6-85b0-ce342c1fcb65.gif)

## Project Creation
### New project 
You can create a new project using a creation wizard which will guide users through each step of the project creation and configuration. Those are illustrated below:

**Project Description**

Every project in Annotation Lab should have the following information:

- a unique name and a short description;
- a team of annotators, reviewers and a manager which will colaborate on the project;
- a configuration which specifies the type of annotations that will be created.

To create a new project, click on the Create Project button on the Home Page and choose a name for it. The project can include a short description and annotation instructions/guidelines.

![Description](https://user-images.githubusercontent.com/46840490/193200891-d2098f36-c300-4693-924d-168801ae5acd.png)

Reserved words cannot be used as project names. The use of keywords like count, permission, or name as project names generated UI glitches. To avoid such issues, these keywords are no longer accepted as project names.

<img class="image image--xl" src="/assets/images/annotation_lab/4.1.0/reserved.gif" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

**Adding Team Members**

When working in teams, projects can be shared with other team members.

The user who creates a project is called a Project Owner. He/She has complete visibility and ownership of the project for its entire lifecycle. If the Project Owner is removed from the user database, then all his/her projects are transfered to a new project owner. The Project Owner can edit the project configuration, can import/export tasks, can create a project team that will work on his project and can access project analytics.
When defining the project team, a project owner has access to three distinct roles: Annotator, Reviewer, and Manager. These are very useful for most of the workflows that our users follow.

- An Annotator is able to see the tasks which have been assigned to him or her and can create annotations on the documents.
- The Reviewer is able to see the work of the annotators and approve it or reject in case he finds issues that need to be solved.
- The Manager is able to see the work of the Annotators and of the Reviewers and he can assign tasks to team members. This is useful for eliminating work overlap and for a better management of the work load.

To add a user to your project team, navigate to the Project Setup page. On the Manage Project Team tab, start typing the name of a user in the available text box. This will populate a list of available users having the username start with the characters you typed. From the dropdown select the user you want to add to your team. Select a role for the user and click on the “Add to team” button.

In the Project Team page users can add/remove/update the team members even in the case of a large number of members. The team members are displayed in a tabular view. Each member has a priority assigned to them for CONLL export which can be changed by dragging them across the list.
![teamMembers](https://user-images.githubusercontent.com/46840490/193060010-3394cccd-93ab-40a8-8479-98155ea8b417.gif)

**Project Configuration**

The Project Configuration itself is a multi-step process. The wizard will guide users through each step and provide information and hints for each step.
![projectConfiguration](https://user-images.githubusercontent.com/46840490/193033349-534cc2ab-2e5a-4caa-a050-0ee650949b21.gif)

### Clone
You can create a copy of a project, by using the Clone option. The option to clone the project is also listed in the kebab menu of each project. The cloned project is differentiated as it contains cloned suffix in its project name.

<img class="image image--xl" src="/assets/images/annotation_lab/4.1.0/dashboardCloneImportExport.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>
  
### Export  
Entire Projects can be exported. The option to export the project is listed in the kebab menu of each project. All project-related items such as tasks, project configuration, project members, task assignments, and comments are included in the export file.

> **NOTE:**  Project export does not contain the model trained in the project as models are independent and not attached to a particular project.
  
### Import  
The project can be imported by uploading the zip in the upload dialog box. When the project is imported back to ALAB, all the item can be seen as it was present when exported. 
  
## Project Grouping
As the number of projects can grow significantly over time, for an easier management and organization of those, the Annotation Lab allows project grouping. As such, it is possible to assign a project to an existing group or to a new group. Each group can be assigned a color which will be used to highlight projects included in that group. Once a project is assigned to a group, the group name proceeds the name of the project. At any time a project can be remove from one group and added to another group.

The list of visible projects can be filtered based on group name, or using the search functionality which applies to both group name and project name. 

<img class="image image--xl" src="/assets/images/annotation_lab/4.1.0/dashboardGroup.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

Projects can be organized in custom groups, and each project card will inherit the group color so that the users can visually distinguish the projects easily in a large cluster of projects. The new color picker for the group is user-friendly and customizable.

![DashboardGroupGIF](https://user-images.githubusercontent.com/46840490/193201637-57a7e7b6-9d25-48b4-9196-e6bed61fa2ad.gif)

## Configuration and Customization 
We currently support multiple predefined project configurations. The most popular ones are Text Classification, Named Entity Recognition (NER) and  Visual NER . Create a setup from scratch or customize a predefined one according to your needs.

For customizing a predefined configuration, click on the corresponding link in the table above and then navigate to the Labeling configuration tab and manually edit or update it to contain the labels you need.

After you finish editing the labels you want to define for your project click the “Save” button.

### Project templates 
We currently support multiple predefined project configurations. The most popular ones are **Text Classification** and **Named Entity Recognition**. Create a setup from scratch or customize a predefined one according to your needs.


 <img class="image image--xl" src="/assets/images/annotation_lab/predefined_configurations.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

 
For customizing a predefined configuration, click on the corresponding link in the table above and then navigate to the Labeling config widget and manually edit/update it to contain the labels you need. 
 
After you finish editing the labels you want to define for your project click the “Save” button. 

### NER Labeling
Named Entity Recognition (NER) refers to the identification and classification of entities mentioned in unstructured text into pre-defined categories such as person names, organizations, locations, medical codes, time expressions, quantities, monetary values, percentages, etc.


The **Annotation Lab** offers support for two types of labels: 
-	Simple labels for NER or assertion models;
-	Binary relations for relation extraction models. 

<img class="image image--xl" src="/assets/images/annotation_lab/labels_def.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>
 
### Assertion Labels 
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

### Relation Extraction 
The Annotation Lab also offers support for relation extraction. Relations are introduced by simply specifying their label in the project configuration. 

```bash
<Relations>
    <Relation value="CancerSize" />
    <Relation value="CancerLocation"/>
    <Relation value="MetastasisLocation"/>
  </Relations>
```
### Constraints for relation labeling

While annotating projects with Relations between Entities, defining constraints (the direction, the domain, the co-domain) of relations is important. Annotation Lab offers a way to define such constraints by editing the Project Configuration. The Project Owner or Project Managers can specify which Relation needs to be bound to which Labels and in which direction. This will hide some Relations in Labeling Page for NER Labels which will simplify the annotation process and will avoid the creation of any incorrect relations in the scope of the project.
To define such constraint, add allowed attribute to the <Relation> tag:
-  L1>L2 means Relation can be created in the direction from Label L1 to Label L2, but not the other way around
-  L1<>L2 means Relation can be created in either direction between Label L1 to Label L2

If the allowed attribute is not present in the tag, there is no such restriction. 

Bellow you can find a sample Project Configuration with Constraints for Relation Labels:

```bash
<View> 
<Header value="Sample Project Configuration for Relations Annotation"/> 
<Relations> 
    <Relation value="Was In" allowed="PERSON>LOC"/> 
    <Relation value="Has Function" allowed="LOC>EVENT,PERSON>MEDICINE"/> 
    <Relation value="Involved In" allowed="PERSON<>EVENT"/> 
    <Relation value="No Constraints"/> 
</Relations> 
<Labels name="label" toName="text"> 
    <Label value="PERSON"/> 
    <Label value="EVENT"/> 
    <Label value="MEDICINE"/> 
    <Label value="LOC"/> 
</Labels> 
<Text name="text" value="$text"/> 
</View>
```

### Visual NER 

Annotating text included in image documents (e.g. scanned documents) is a common use case in many verticals but comes with several challenges. With the new Visual NER Labeling config, we aim to ease the work of annotators by allowing them to simply select text from an image and assign the corresponding label to it.

This feature is powered by Spark OCR 3.5.0; thus a valid Spark OCR license is required to get access to it.

Here is how this can be used:
1.  Upload a valid Spark OCR license. See how to do this [here](https://nlp.johnsnowlabs.com/docs/en/alab/byol).
2.  Create a new project, specify a name for your project, add team members if necessary, and from the list of predefined templates (Default Project Configs) choose “Visual NER Labeling”.
3.  Update the configuration if necessary. This might be useful if you want to use other labels than the currently defined ones. Click the save button. While saving the project, a confirmation dialog is displayed to let you know that the Spark OCR pipeline for Visual NER is being deployed.
4.  Import the tasks you want to annotate (images).
5.  Start annotating text on top of the image by clicking on the text tokens or by drawing bounding boxes on top of chunks or image areas.
6.  Export annotations in your preferred format.

The entire process is illustrated below: 

<img class="image image--xl" src="/assets/images/annotation_lab/2.1.0/invoice_annotation.gif" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

 
### Import Tasks
Once a new project is created and its configuration is saved, the user is redirected to the **Import page**. Here the user has multiple options for importing tasks.

<img class="image image--xl" src="/assets/images/annotation_lab/1.6.0/import.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

### Plain text file 
When you upload a plain text file, only one task will be created which will contain the entire data in the input file.

This is an update from versions of Annotation Lab when the input text file was split by the new line character and one task was created for each line. 
{:.warning}

### JSON file
For bulk importing a list of documents you can use the json import option. The expected format is illustrated in the image below. It consists of a list of dictionaries, each with 2 keys-values pairs  (“text” and “title”). 

```bash
[{  "text": "Task text content.", "title":"Task title"}]
```


### CSV, TSV file
When CSV / TSV formatted text file is used, column names are interpreted as task data keys:

```bash
Task text content, Task title
this is a first task, Colon Cancer.txt
this is a second task, Breast radiation therapy.txt
```

### Import annotated tasks

When importing tasks that already contain annotations (e.g. exported from another project, with predictions generated by pre-trained models) the user has the option to	overwrite completions/predictions or to skip the tasks that are already imported into the project. 
<img class="image image--xl" src="/assets/images/annotation_lab/1.6.0/overwrite.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

### Dynamic Task Pagination

The support for pagination offered by earlier versions of the Annotation Lab involved the use of the `<pagebreak>` tag. A document pre-processing step was necessary for adding/changing the page breaks and those involved extra effort from the part of the users. 

Annotation Lab 2.8.0 introduces a paradigm change for pagination. Going forward, pagination is dynamic and can be configured according to the user’s needs and preferences from the Labeling page. Annotators or reviewers can now choose the number of words to include on a single page from a predefined list of values or can add the desired counts.

<img class="image image--xl" src="/assets/images/annotation_lab/2.8.0/158536371-10ce44f0-4b22-4306-975e-42ce0191692d.gif" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

A new settings option has been added to prevent splitting a sentence into two different pages.

<img class="image image--xl" src="/assets/images/annotation_lab/2.8.0/158552636-1b9f8814-5e05-4904-8ab4-401ea476d32e.png" style="width:600%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>
 

## Export Annotations 
The completions and predictions are stored in a database for fast search and access. Completions and predictions can be exported into the formats described below.

### JSON
You can convert and export the completions and predictions to json format by using the JSON option on the **Export** page. 
The obtained format is the following:

```bash
[
  {
    "completions": [],
    "predictions": [
      {
        "created_username": "SparkNLP Pre-annotation",
        "result": [
          {
            "from_name": "label",
            "id": "7HGzTLkNUA",
            "source": "$text",
            "to_name": "text",
            "type": "labels",
            "value": {
              "end": 3554,
              "labels": [
                "Symptom_Name"
              ],
              "start": 3548,
              "text": "snores"
            }
          }
        ],
        "created_ago": "2020-11-09T14:44:57.713743Z",
        "id": 2001
      }
    ],
    "created_at": "2020-11-09 14:41:39",
    "created_by": "admin",
    "data": {
      "text": "Cardiovascular / Pulmonary\nSample Name: Angina - Consult\nDescription: Patient had a recurrent left arm pain after her stent, three days ago, and this persisted after two sublingual nitroglycerin.\n(Medical Transcription Sample Report)\nHISTORY OF PRESENT ILLNESS: The patient is a 68-year-old woman whom I have been following, who has had angina. In any case today, she called me because she had a recurrent left arm pain after her stent, three days ago, and this persisted after two sublingual nitroglycerin when I spoke to her.",
      "title": "sample_document3.txt",
      "pre_annotation": true
    },
    "id": 2
  }]
```
### CSV
Results are stored in comma-separated tabular file with column names specified by "from_name" "to_name" values

### TSV
Results are stored in tab-separated tabular file with column names specified by "from_name" "to_name" values

### CONLL2003

The CONLL export feature generates a single output file, containing all available completios for all the tasks in the project. The resulting file has the following format: 
```bash
-DOCSTART- -X- O
Sample -X- _ O
Type -X- _ O
Medical -X- _ O
Specialty: -X- _ O
Endocrinology -X- _ O

Sample -X- _ O
Name: -X- _ O
Diabetes -X- _ B-Diagnosis
Mellitus -X- _ I-Diagnosis
Followup -X- _ O

Description: -X- _ O
Return -X- _ O
visit -X- _ O
to -X- _ O
the -X- _ O
endocrine -X- _ O
clinic -X- _ O
for -X- _ O
followup -X- _ O
management -X- _ O
of -X- _ O
type -X- _ O
1 -X- _ O
diabetes -X- _ O
mellitus -X- _ O
Plan -X- _ O
today -X- _ O
is -X- _ O
to -X- _ O
make -X- _ O
adjustments -X- _ O
to -X- _ O
her -X- _ O
pump -X- _ O
based -X- _ O
on -X- _ O
a -X- _ O
total -X- _ O
daily -X- _ B-FREQUENCY
dose -X- _ O
of -X- _ O
90 -X- _ O
units -X- _ O
of -X- _ O
insulin -X- _ O
…
```

User can specify if only starred completions should be included in the output file by checking "Only ground truth" option before generating the export.

### Allow the export of tasks without completions

Previous versions of the Annotation Lab only allowed the export of tasks that contained completions. From version 2.8.0 on, the tasks without any completions can be exported as this can be necessary for cloning projects. In the case where only tasks with completions are required in the export, users can enable the “Exclude tasks without Completions” option on the export page. 

 ![export-page](https://user-images.githubusercontent.com/26042994/154637982-55872de3-85e2-4aaf-be4d-7e8c1d59417d.png)


