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

<img class="image image--xl" src="/assets/images/annotation_lab/image003.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>


In projects with more than one annotator, the option **Show completions to all users** should be selected only if each team member is allowed to have access to the completions of the other annotators.


## Share your project with the annotation team
When working in teams, projects can be shared with other team members. Fine grained access right can be  assigned to each team member. For example, when working on confidential documents annotation, the options **Create**, **View**, **Update** and **Delete** should be assigned to annotators, while data scientists should also be able to **Import** and **Export** data.

 <img class="image image--xl" src="/assets/images/annotation_lab/image005.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

## Supported Project Types
We currently support multiple predefined project configurations. The most popular ones are **Text Classification** and **Named Entity Recognition**. Create a setup from scratch or customize a predefined one according to your needs.


 <img class="image image--xl" src="/assets/images/annotation_lab/predefined_configurations.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

 
For customizing a predefined configuration, click on the corresponding link in the table above and then navigate to the Labeling config widget and manually edit/update it to contain the labels you need. 
 
After you finish editing the labels you want to define for your project click the “Save” button. 
 
## Text Classification Project

The Annotation Lab offers two typed of classification widgets:

- The first one supports single choice labels. You can activate it by choosing **Text Classification** from the list of predefined projects. The labels can be changed by directly editing them in the **Labeling Config** XML style widget. The updates will be automatically reflected in the right side preview. 

<img class="image image--xl" src="/assets/images/annotation_lab/sent_analysis.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>
- The second configuration offers support for multi-class classification. It can be activated by clicking on the **Multi classification** link in the list of predefined configurations. This option will add to the labeling config widget multiple checkboxes, grouped by headers. The names of the choices and well as the headers are customizable. You can also add new choices if necessary. 

<img class="image image--xl" src="/assets/images/annotation_lab/image013.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

 
## Named Entity Recognition Project

Named entity recognition refers to the identification and classification of entities mentioned in unstructured text into pre-defined categories such as person names, organizations, locations, medical codes, time expressions, quantities, monetary values, percentages, etc.


The **Annotation Lab** offers support for two types of labels: 
-	Simple labels for NER or assertion models;
-	Binary relations for relation extraction models. 

<img class="image image--xl" src="/assets/images/annotation_lab/labels_def.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

Labels customization:
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

The Annotation Lab also offers support for relation extraction. Relations are introduced by simply specifying their label. 

```bash
<Relations>
    <Relation value="CancerSize" />
    <Relation value="CancerLocation"/>
    <Relation value="MetastasisLocation"/>
  </Relations>
```
No other constraints can currently be enforced on the labels linked by the defined relations so the annotators must be extra careful and follow the annotation guidelines that specify how the defined relations can be used.  
 
## Preannotations with Spark NLP 

The **Annotation Lab** offers out-of-the-box support for **NER Pre-annotations**. Those are extremely useful for bootstraping any NER project, as the annotation team does not start the labeling from scratch but can leverage the existing knowledge transfer from domain experts to models. This way, the annotation efforts are significantly reduced.


On the project setup screen you can find a Spark NLP pipeline config widget which lists all available models together with the labels those are predicting. By simply selecting the relevant labels for your project and clicking the add button you can add the predefined labels to your project and take advantage of the Spark NLP auto labeling capabilities. 


In the below example we are reusing the posology model that comes with 7 labels related to drugs.  
<img class="image image--xl" src="/assets/images/annotation_lab/spark_nlp_models.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>
