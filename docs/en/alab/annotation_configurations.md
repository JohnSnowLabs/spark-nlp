---
layout: docs
comment: no
header: true
seotitle: Annotation Lab | John Snow Labs
title: Configurations
permalink: /docs/en/alab/annotation_configurations
key: docs-training
modify_date: "2022-12-11"
use_language_switcher: "Python-Scala"
show_nav: true
sidebar:
  nav: annotation-lab
---

<style>
bl {
  font-weight: 400;
}

es {
  font-weight: 400;
  font-style: italic;
}
</style>

## Simplified workflow

### Direct Submit

Using the classical annotation workflow, when an annotator works on a task, a series of actions are necessary for creating a new annotation and submitting it as ground truth:

1. Create the completion
2. Save the completion
3. Submit the completion
4. Confirm submission
5. Load next task

This process is adapted for more complex workflows and large tasks. For simple projects with smaller tasks, Annotation Lab now offers a simplified workflow. Annotators can submit a completion with just one click.

The Project Owner/Manager can activate this option from the <es>Settings</es> dialog (Customize Labels) in the <es>Configuration</es> step of the <es>Setup</es> page. Once enabled, annotators can see the submit button on the labeling page. A second option is available on the same dialog for Project Owner/Manager: _Serve next task after completion submission_. Once enabled, annotators can see the next task on the labeling page after submitting the completion for the current task.

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/direct_submit.gif" style="width:100%;"/>

> **Note:**
>
> 1. Annotator can _Save/Update_ completion using `CTRL+Enter`
> 2. Annotator can _Submit_ completion using `ALT+Enter`

<br />

### Accept Prediction

When predictions are available for a task, Annotator can accept the predictions with just one click and navigate automatically to the next task. When users click on Accept Prediction, a new completion is created based on the prediction, then submitted as ground truth, and the next task in line (assigned to the current annotator/reviewer and with <es>Incomplete</es> or <es>In Progress</es> status) is automatically served.

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/accept_prediction.gif" style="width:100%;"/>

> **NOTE:** Press backspace key (on windows) or delete key (on mac) to delete the selected relation from the labeling editor or use the delete action icon on the Relations widget.

## Labeling editor Settings

The labeling editor offers some configurable features. For example, you can modify the editor's layout, show or hide predictions, annotations, or the confidence panel, show or hide various controls and information. It is also possible to keep a label selected after creating a region, display labels on bounding boxes, polygons and other regions while labeling, and show line numbers for text labeling.

### Enable labeling hotkeys

This option enables/disable the hotkeys assigned to taxonomy labels to use the hotkeys during the annotation process.

<br />

### Show hotkey tooltips

This option shows/hides the hotkey and tooltip on the taxonomy label and the control buttons. _Enable labeling hotkeys_ must be enabled for this option to work.

<img class="image image__shadow image__align--center" src="/assets/images/annotation_lab/4.1.0/toggle_show_hotkey_tooltips.png" style="width:50%;"/>

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/show_hotkey_tooltips.png" style="width:100%;"/>

<br />

### Show labels inside the regions

When you enable this option, the labels assigned to each annotated region are displayed on the respective region.

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/show_labels_inside_regions.png" style="width:100%;"/>

<br />

### Keep label selected after creating a region

This option helps users quickly annotate sequences of the same label by keeping the label selected after the annotation of a region.

With the option unchecked:

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/keep_label_selected_off.gif" style="width:100%;"/>

With the option checked:

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/keep_label_selected_on.gif" style="width:100%;"/>

<br />

### Select regions after creating

This option keeps the annotated region selected after annotation. In this way, it will be easier for users to quickly change the assigned label for the last selected region if necessary.

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/select_region_after_creating.gif" style="width:100%;"/>

<br />

### Show line numbers for Text

This option adds line numbers to the text content to annotate in the labeling editor.

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/toggle_line_number.png" style="width:100%;"/>

<br />

### Label all occurrences of selected text

When checked, this option allow users to annotate all occurences of a text in the current task in one step.

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/label_all_occurence.gif" style="width:100%;"/>

## Labeling editor Customizations

The Labeling editor is highly customizable. Project Owners and Managers can change the layout of their projects based on their needs.

### Search filter for a large number of labels

When a project has a large number of NER/Assertion labels in the taxonomy, the display of the taxonomy takes a lot of screen space, and it is difficult for annotators to navigate through all labels. To tackle this challenge, Annotation Lab supports search for labels in NER projects (an autocomplete search option).

To add the search bar for NER Labels or Choices, use the `Filter` tag as shown in the following XML configuration.

```
<Filter />

<View>
*** enclose labels tags here ***
</View>

<View>
*** enclose text tags here ***
</View>
```

**Parameters:**

The following parameters/attributes can be used within the `Filter` tag.

| Param       | Type   | Default      | Description                                    |
| ----------- | ------ | ------------ | ---------------------------------------------- |
| placeholder | string | Quick Filter | Placeholder text for filter                    |
| minlength   | number | 3            | Size of the filter                             |
| style       | string |              | CSS style of the string                        |
| hotkey      | string |              | Hotkey to use to focus on the filter text area |

**Usage Example:**

```
<Filter placeholder="Quick Filter"/>
```

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/filter.png" style="width:100%;"/>

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

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/ner_label_search.gif" style="width:100%;"/>

<br />

### Resizable label and text container

While annotating longer text documents annotators may need to scroll to the top of the document for selecting the label to use, and then scroll down to create a label. Also, if the text is large, annotators have to scroll to a certain section because the textbox size is fixed. In those cases, the annotation experience can be improved by creating a scrollable labeling area and textbox area.

To add the scroll bar, the `View` tag with a fixed height and `overflow-y:scroll` style property can be used as shown in the following XML config structure:

```
<View style="background:white; height: 100px; overflow-y:scroll; resize:vertical; position:sticky; top:0;">
*** enclose labels tags here ***
</View>

<View style="resize:vertical; margin-top:10px; max-height:400px; overflow-y:scroll;">
**** enclose text tags here**
</View>
```

Once it has been added and saved to the Project Configuration, the scroll bar should be visible.

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/labeling_scroll_bar.png" style="width:100%;"/>

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

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/labeling_scroll_bar.gif" style="width:100%;"/>

## Toggle Preview Window

Label configuration editor and <es>Preview Window</es> covers 50/50 part of the screen. It can make editing larger XML configurations difficult. For a better editing experience, we can use the Toggle Preview Window button to have the editor use full screen width.

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/toggle_preview_window.gif" style="width:100%;"/>

## Switch Role

For users having multiple roles (_Annotator/Reviewer/Manager_) the labeling page can get confusing. Switch Role filter present on the top-right corner can help address this problem. This filter was introduced in Annotation Lab from version <bl>2.6.0</bl>, previously refered to as <es>View As</es> filter. When selecting _Annotator_ option, the view changes to facilitate annotating the task. Similar changes to the view applies when switching to _Reviewer_ or _Manager_ option. The selection persists even when the tab is closed or refreshed.

<img class="image image__shadow image__align--center" src="/assets/images/annotation_lab/4.1.0/switch_role.png" style="width:40%;"/>
