---
layout: docs
comment: no
header: true
seotitle: Annotation Lab | John Snow Labs
title: Annotation Page Customization
permalink: /docs/en/alab/annotation_screen_customization
key: docs-training
modify_date: "2022-01-28"
use_language_switcher: "Python-Scala"
show_nav: true
sidebar:
    nav: annotation-lab
---

The `Annotation Page` is highly customizable. Project Owners and Managers can change the layout of their projects based on their needs. 

## Search filter for a large number of labels


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


## Resizable labels area and textbox area 

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
