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