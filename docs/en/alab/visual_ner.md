---
layout: docs
comment: no
header: true
seotitle: Visual NER | John Snow Labs
title: Visual NER
permalink: /docs/en/alab/visual_ner
key: docs-training
modify_date: "2021-11-11"
use_language_switcher: "Python-Scala"
show_nav: true
sidebar:
    nav: annotation-lab
---

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

## Support for multi-page PDF documents

When a valid Saprk OCR license is available, Annotation Lab offers support for multi-page PDF annotation. The complete flow of import, annotation, and export for multi-page PDF files is currently supported.

Users have two options for importing a new PDF file into the Visual NER project
- Import PDF file from local storage;
- Add a link to the PDF file in the file attribute.

<img class="image image--xl" src="/assets/images/annotation_lab/2.3.0/import_pdf.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

After import, the task becomes available on the `Tasks Page`. The title of the new task is the name of the imported file. 

<img class="image image--xl" src="/assets/images/annotation_lab/2.3.0/import_pdf_2.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

On the labeling page, the PDF file is displayed with pagination so that annotators can annotate on the PDF document one page at a time.