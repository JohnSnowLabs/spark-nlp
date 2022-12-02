---
layout: docs
comment: no
header: true
seotitle: Annotation Lab | John Snow Labs
title: Project Configuration
permalink: /docs/en/alab/project_configuration
key: docs-training
modify_date: "2022-11-01"
use_language_switcher: "Python-Scala"
show_nav: true
sidebar:
  nav: annotation-lab
---

Annotation Lab currently supports multiple predefined project configurations. The most popular ones are `Text Classification`, `Named Entity Recognition` (NER) and `Visual NER`. Create a setup from scratch or customize a predefined one according to your needs.

For customizing a predefined configuration, click on the corresponding link in the table above and then navigate to the Labeling configuration tab and manually edit or update it to contain the labels you want.

After you finish editing the labels you want to define for your project click the “Save” button.

### Project templates

We currently support multiple predefined project configurations. The most popular ones are **Text Classification**, **Named Entity Recognition** and **Visual NER**.

**Content Type**

The first step when creating a new project or customizing an existing one is to choose what content you need to annotate. Five content types are currently supported: `Audio`, `HTML`, `Image`, `PDF` and `Text`. For each content type a list of available templates is available. You can pick any one of those as a starting point in your project configuration.

 <img class="image image--xl image__shadow" src="/assets/images/annotation_lab/4.2.0/template.png" style="width:100%"/>

For customizing a predefined configuration, choose a `Content Type` and then a template from the list. Then navigate to the `Customize Labels` tab and manually edit/update the configuration to contain the labels you need.

Users can add custom labels and choices in the project configuration from the Visual tab for both text and Visual NER projects.

<img class="image image--xl image__shadow" src="https://user-images.githubusercontent.com/33893292/203908059-95b96598-ef34-4418-8b9a-48dd6b03b409.gif" style="width:100%;"/>

After you finish editing the labels click the “Save” button.

### Named Entity Recognition

Named Entity Recognition (NER) refers to the identification and classification of entities mentioned in unstructured text into pre-defined categories such as person names, organizations, locations, medical codes, time expressions, quantities, monetary values, percentages, etc.

The **Annotation Lab** offers support for two types of labels:

- Simple labels for NER or assertion models;
- Binary relations for relation extraction models.

<img class="image image--xl image__shadow" src="/assets/images/annotation_lab/4.2.0/NER_Template.png" style="width:100%;"/>

<br />
 
### Assertion Labels 
The syntax for defining an Assertion Status label is the same as for the NER labels, with an additional attribute - **assertion** which should be set to true (see example below). This convention is defined by Annotation Lab users which we exploited for identifying the labels to include in the training and prediction of Assertion Models.
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

> **NOTE:** Notice assertion="true" in **Absent** and **Past** labels, which marks each of those labels as Assertion Status Labels.

<br />

### Classification

The choices tag is used as part of the classification projects to create a group of choices. It can be used for a single or multiple-class classification. According to the parameters used along with the choices tag, annotators can select single or multiple choices.

**Parameters**

The Choices tag supports the following parameters/attributes:

| Param           | Type                   | Default  | Description                                                                              |
| --------------- | ---------------------- | -------- | ---------------------------------------------------------------------------------------- |
| required        | `boolean`              | `false`  | Verify if a choice is selected                                                           |
| requiredMessage | `string`               |          | Show a message if the required validation fails                                          |
| choice          | `single` \| `multiple` | `single` | Allow user to select single or multiple answer                                           |
| showInline      | `boolean`              | `false`  | Show choices in a single visual line                                                     |
| perRegion       | `boolean`              |          | Use this attribute to select an option for a specific region rather than the entire task |

```bash
<!--text classification labeling config-->
<View>
  <Text name="text" value="$text"/>
  <Choices name="surprise" toName="text" choice="single" required='true' requiredMessage='Please select choice'>
    <Choice value="surprise"/>
    <Choice value="sadness"/>
    <Choice value="fear"/>
    <Choice value="joy"/>
  </Choices>
</View>
```

![choices-example](https://user-images.githubusercontent.com/45035063/196096241-c155f4cb-df5c-4b1c-a19b-0a5e349a756e.gif)

When using the `perRegion` attribute, choices can be defined for each chunk annotation as shown below:

![perRegion](https://user-images.githubusercontent.com/45035063/197151637-fa9034b1-0d09-4658-b1bb-7c47e68f435c.gif)

<br />

### Relation Extraction

Annotation Lab also offers support for relation extraction. Relations are introduced by simply specifying their label in the project configuration.

```bash
<Relations>
    <Relation value="CancerSize" />
    <Relation value="CancerLocation"/>
    <Relation value="MetastasisLocation"/>
  </Relations>
```

<br />

### Constraints for relation labeling

While annotating projects with Relations between Entities, defining constraints (the direction, the domain, the co-domain) of relations is important. Annotation Lab offers a way to define such constraints by editing the Project Configuration. The Project Owner or Project Managers can specify which Relation needs to be bound to which Labels and in which direction. This will hide some Relations in Labeling Page for NER Labels which will simplify the annotation process and will avoid the creation of any incorrect relations in the scope of the project.
To define such constraint, add allowed attribute to the <Relation> tag:

- L1>L2 means Relation can be created in the direction from Label L1 to Label L2, but not the other way around
- L1<>L2 means Relation can be created in either direction between Label L1 to Label L2

If the allowed attribute is not present in the tag, there is no such restriction.

Below you can find a sample Project Configuration with constraints for Relation Labels:

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
