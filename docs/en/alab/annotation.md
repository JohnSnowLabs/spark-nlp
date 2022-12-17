---
layout: docs
comment: no
header: true
seotitle: Annotation Lab | John Snow Labs
title: Manual Annotation
permalink: /docs/en/alab/annotation
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

The Annotation Lab keeps a human expert as productive as possible. It minimizes the number of mouse clicks, keystrokes, and eye movements in the main workflow. The continuous improvement in the UI and the UX is from iterative feedback from the users.

Annotation Lab supports keyboard shortcuts for all types of annotations. It enables having one hand on the keyboard, one hand on the mouse, and both eyes on the screen at all times. One-click completion and automatic switching to the next task keep experts in the loop.

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/annotation_main.png" style="width:100%;"/>

On the header of the Labeling area, you can find the list of labels defined for the project. In the center, it displays the content of the task. On the right, there are several widgets categorized into different groups.

- Annotations
- Versions
- Progress

## Labeling Widgets

### Completions

A completion is a list of annotations manually defined by a user for a given task. After completing annotation on a task (e.g., all entities highlighted in the text, or one or more classes is assigned to the task in the case of classification projects) user clicks on the `Save` button to save their progress or `Submit` button to submit the completion.

A submitted completion is no longer editable, and the user cannot delete it. Creating a new copy of the submitted completion is the only option to edit it. An annotator can modify or delete their completions only if completions are not submitted yet.

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/completion_submit.png" style="width:100%;"/>

Dedicated action icons are available on the completions widgets to allow users to quickly run actions like delete, copy, set ground-truth.

<img class="image image__shadow" src="https://user-images.githubusercontent.com/45035063/204707502-acaba8b8-43d0-4354-8c14-4f66132131ad.png" style="width:100%;"/>

It is an important to ensure a complete audit trail of all user actions. Annotation Lab tracks the history and details of any deleted completions. It means it is possible to see the name of the completion creator, date of creation, and deletion.

<img class="image image__shadow image__align--center" src="/assets/images/annotation_lab/4.1.0/completion_history.png" style="width:40%;"/>

<br />

### Predictions

A prediction is a list of annotations created automatically by <bl>Spark NLP</bl> pre-trained [models](https://nlp.johnsnowlabs.com/docs/en/alab/models) or from the [rules](https://nlp.johnsnowlabs.com/docs/en/alab/rules). A project owner/manager can create predictions using the `Pre-Annotate` button from the <es>Tasks</es> page. Predictions are read-only, which means users can see the predictions but cannot modify those.

To reuse a prediction to bootstrap the annotation process, users can copy it to a new completion. This new completion bootstrapped from the prediction is editable.

<img class="image image__shadow image__align--center" src="/assets/images/annotation_lab/4.1.0/prediction.png" style="width:40%;"/>

<br />

### Confidence

From version <bl>3.3.0</bl>, running pre-annotations on a text project provides one extra piece of information for the automatic annotations - the confidence score. This score shows the confidence the model has for each of the labeled chunks it predicts. It is calculated based on the benchmarking information of the model used to pre-annotate and the score of each prediction. The confidence score is available when working on <es>Named Entity Recognition</es>, <es>Relation Extraction</es>, <es>Assertion</es>, and <es>Classification</es> projects and is also generated when using [Rules](https://nlp.johnsnowlabs.com/docs/en/alab/rules).

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/confidence.png" style="width:100%;"/>

On the Labeling page, when selecting the <es>Prediction</es> widget, users can see all preannotation in the <es>Annotations</es> section with a score assigned to them. Using the confidence slider, users can filter out low confidence labels before starting to edit/correct the labels. Both _Accept Prediction_ and _Add a new completion based on this prediction_ operation apply to the filtered annotations from the confidence slider.

<br />

### Annotations

The Annotations widget has two sections.

<bl>Regions</bl> - Gives a list overview of all annotated chunks. When you click on any annotation, it gets automatically highlighted in the labeling editor. We can edit or remove annotations from here.

<bl>Relations</bl> - Lists all the relations that have been created. When the user moves the mouse over any one relation, it is highlighted in the labeling editor.

<br />

### Progress

Annotator/Reviewer can see their overall work progress from within the labeling page. The status is calculated for their assigned work.

<bl>For Annotator View:</bl>

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/progress_annotator.png" style="width:40%;"/>

<bl>For Reviewer View:</bl>

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/progress_reviewer.png" style="width:40%;"/>

## Text Annotation

### Named Entity Recognition

To extract information using NER labels, we first click on the label to select it or press the shortcut key assigned to it, and then, with the mouse, select the relevant part of the text. We can easily edit the incorrect labeling by clicking on the labeled text and then selecting the new label you want to assign to this text.

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/add_label.png" style="width:100%;"/>

To delete the label from the text, we first click on the text on the labeling editor and then press backspace.

#### Trim leading and ending special characters in annotated chunks

When annotating text, it is possible and probable that the annotation is not very precise and the chunks contain leading/trailing spaces and punctuation marks. By default all the leading/trailing spaces and punctuation marks are excluded from the annotated chunk. The labeling editor settings has a new configuration option that can be used to enable/disable this feature if necessary.

![trim_spaces_punctuations](/assets/images/annotation_lab/4.1.0/trimming_characters.gif)

<br />

### Assertion Labels

To add an assertion label to an extracted entity, select the assertion label and select the labeled entity (from NER) in the labeling editor. After this, the extracted entity will have two labels - one for NER and one for assertion. In the example below, the chunks <es>heart disease</es>, <es>kidney disease</es>, <es>stroke</es> etc., were extracted first using the NER label - _Symptom_ (pink color) and then the assertion label - _Absent_ (green color).

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/assertion.png" style="width:100%;"/>

<br />

### Relation Extraction

Creating relations with the Annotation Lab is very simple. First, click on any one labeled entity, then press the <es>r</es> key and click on the second labeled entity.

<img class="image image__shadow" src="/assets/images/annotation_lab/4.2.0/rel_ann.gif" style="width:100%;"/>

You can add a label to the relation, change its direction or delete it using the contextual menu displayed next to the relation arrow or from the relation box.

<img class="image image__shadow" src="/assets/images/annotation_lab/relations2.png" style="width:40%;"/>

<br />

#### Cross page Annotation

From version <bl>2.8.0</bl>, Annotation Lab supports cross-page NER annotation for <es>Text</es> projects. It means that Annotators can annotate a chunk starting at the bottom of one page and finishing on the next page. This feature is also available for <es>Relations</es>. Previously, relations were created between chunks located on the same page. But now, relations can be created among tokens located on different pages. The way to do this is to first [change the pagination settings](/docs/en/alab/import#dynamic-task-pagination) to include the tokens to be linked on the same page, then create the relation annotation between the tokens and finally go back to the original pagination settings. The annotation is presented through connectors after updating the pagination.

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/crosspage_annotation.gif" style="width:100%;"/>

<br />

## Visual NER Annotation

Annotating text included in image documents (e.g., scanned documents) is a common use case in many verticals but comes with several challenges. With the <es>Visual NER</es> labeling config, we aim to ease the work of annotators by allowing them to select text from an image and assign the corresponding label to it.

This feature is powered by <bl>Visual NLP</bl> library; hence a valid [Visual NLP](/docs/en/ocr) license is required to get access to it.

Here is how we can use it:

1. Upload a valid <bl>[Visual NLP](/docs/en/ocr)</bl> license. See how to do this [here](/docs/en/alab/byol).
2. Create a new project, specify a name for your project, add team members if necessary, and from the list of predefined templates (Default Project Configs) choose `Visual NER Labeling` under **IMAGE** content type.
3. Update the configuration if necessary. This might be useful if you want to use other labels than the default ones. Click the `Save Config` button. While saving the project, a confirmation dialog is displayed to ask if you want to deploy the OCR pipeline. Select `Yes` from the confirmation dialog.
4. Import the tasks you want to annotate (images or PDF documents).
5. Start annotating text on top of the image by clicking on the text tokens, or by drawing bounding boxes on top of chunks or image areas.
6. Export annotations in your preferred format.

The entire process is illustrated below:

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/visual_ner.gif" style="width:100%;"/>

### Support for multi-page PDF documents

When a valid Visual NLP license is available, Annotation Lab offers support for multi-page PDF annotation. We can import, annotate, and export multi-page PDF files easily.

Users have two options for importing a new PDF file into the Visual NER project:

1. Import PDF file from local storage.
2. Add a link to the PDF file in the file attribute.

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/import_pdf_from_link.png" style="width:100%;"/>

After import, the task becomes available on the <es>Tasks</es> Page. The title of the new task is the name of the imported file. On the labeling page, the PDF viewer has pagination support so that annotators can annotate on the PDF document one page at a time.

Users can also jump to a specific page in multi-page task, instead of passing through all pages to reach a target section of a PDF document.

<img class="image image__shadow" src="https://user-images.githubusercontent.com/26042994/203706994-ebb86f14-0a9c-4633-a4c9-8873ae613acb.gif" style="width:100%;"/>

<br />

### Support for multiple OCR servers

Just like for [Preannotation](/docs/en/alab/preannotation) servers, Annotation Lab supports deployment of multiple OCR servers. If a user has uploaded a [Visual NLP](/docs/en/ocr) license, <es>OCR inference</es> is enabled.

To work on a Visual NER project, users have to deploy at least one OCR server. Any OCR server can perform preannotation. To select the OCR server, users need to go to the <es>Import</es> page, click on the OCR Server button on the top-right corner and from the popup, choose one of the available OCR servers. If no suitable OCR server is present, you can create a new server by selecting the `Create Server` option and then clicking on the `Deploy` button.

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/ocr_server.gif" style="width:100%;"/>
