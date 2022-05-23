---
layout: docs
comment: no
header: true
seotitle: Annotation Lab | John Snow Labs
title: Annotation Settings
permalink: /docs/en/alab/settings
key: docs-training
modify_date: "2021-10-06"
use_language_switcher: "Python-Scala"
show_nav: true
sidebar:
    nav: annotation-lab
---

The labeling screen offers some configurable features. For example, you can modify the layout of the screen, hide or show predictions, annotations, or the results panel, hide or show various controls and buttons. It is also possible to keep a label selected after creating a region, display labels on bounding boxes, polygons and other regions while labeling, and show line numbers for text labeling.


## Enable labeling hotkeys

This option specifies if the hotkeys assigned to taxonomy labels should be used for activating them during the annotation process.
<img class="image image--xl" src="/assets/images/annotation_lab/settings/Enable_labeling_hotkeys.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

## Show hotkey tooltips

When the enable labelling hotkeys is turned on, it is possible to also show the hotkey tooltips on some of the buttons - the ones which have hotkeys assigned. 
<img class="image image--xl" src="/assets/images/annotation_lab/settings/settings_hotkey.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

<img class="image image--xl" src="/assets/images/annotation_lab/settings/hotkeys.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

## Show labels hotkey tooltips

When this option is checked tooltips will be displayed on mouse over for the buttons that have hotkeys assigned. 
<img class="image image--xl" src="/assets/images/annotation_lab/settings/tooltips.gif" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>


## Show labels inside the regions

When this option is checked the labels assigned to each annotated region are displayed on the respective region as shown in the image below. 
<img class="image image--xl" src="/assets/images/annotation_lab/settings/labels_inside_regions.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

## Keep label selected after creating a region

This option helps users quickly annotate sequences of the same label by keeping the label selected after the annotation of a region. 
With the option unchecked:
<img class="image image--xl" src="/assets/images/annotation_lab/settings/region_not_selected.gif" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>
With the option checked:
<img class="image image--xl" src="/assets/images/annotation_lab/settings/region_selected.gif" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

## Select regions after creating

This option keeps the annotated region selected after annotation. In this way, it will be easier for users to quickly change the assigned label for the last selected region if necessary. 
<img class="image image--xl" src="/assets/images/annotation_lab/settings/annotations_selected.gif" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

## Show line numbers for Text

This option adds line counts for the text to annotate as illustrated below. 
<img class="image image--xl" src="/assets/images/annotation_lab/settings/show_line_number.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

## Label all occurrences of selected text

When checked, this option allow users to annotate all occurences of a label in the current task in one step. 
<img class="image image--xl" src="/assets/images/annotation_lab/settings/all_occ.gif" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

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


### Project Configuration Box

Project Config XML window and Preview window used to cover 50/50 portion of the screen, which made it tough to edit larger XML configs. We have introduced a Full Screen Toggle Button which makes editing configs much easier.

 ![improved_project_configuration_box](https://user-images.githubusercontent.com/26042994/158370541-0a99c9f7-d9c9-4002-8203-c3dc16ed0e16.gif)
