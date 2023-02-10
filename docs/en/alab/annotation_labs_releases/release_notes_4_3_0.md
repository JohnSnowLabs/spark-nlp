---
layout: docs
header: true
seotitle: Annotation Lab | John Snow Labs
title: Annotation Lab Release Notes 4.3.0
permalink: /docs/en/alab/annotation_labs_releases/release_notes_4_3_0
key: docs-licensed-release-notes
modify_date: "2022-12-07"
show_nav: true
sidebar:
  nav: annotation-lab
---

<div class="h3-box" markdown="1">

## 4.3.0

Release date: **25-11-2022**

Annotation Lab 4.3.0 adds support for Finance and Legal NLP Libraries, Finance and Legal License Scopes, and access to pre-trained Visual NER models on the Models Hub. It also allows easy task import directly from S3 and keeps the complete history of training logs. The release also includes stabilization and fixes for several issues reported by our user community.

Here are the highlights of this release:

### Highlights

- **Support for Finance NLP and Legal NLP**. Annotation Labs now includes a full-fledged integration with two new NLP libraries: Finance NLP and Legal NLP. Pretrained models for the Finance and Legal verticals are now available on the Models Hub page, covering tasks such as Entity Recognition, Assertion Status, and Text Classification.
- **Searching models on Models Hub**. A new filter named Edition was added to the Models Hub. It includes all supported NLP editions: Healthcare, Opensource, Legal, Finance, and Visual. It will ease search for models specific to an Edition, which can then easily be downloaded and used within Annotation Lab projects.
- **Support for Finance and Legal Licenses**. Annotation Lab now supports import of licenses with legal and/or finance scopes. It can be uploaded from the Licenses page. Similar to Healthcare and Visual licenses, they unlock access to optimized annotators, models, embeddings, and rules.
- **Pre-annotations using Finance and Legal models**. Finance and Legal models downloaded from the Models Hub can be used for pre-annotation in NER, assertion status, and classification projects.
- **Train Finance and Legal models**. Two new options: Legal and Finance libraries were added for selection when training a new NER model in Annotation Lab. The new options are only available when at least one valid license with the corresponding scope is added to the License page.
- **Import tasks from S3**. Annotation Lab now supports importing tasks/documents stored on Amazon S3. In the Import Page, a new section was added which allows users to define S3 connection details. All documents in the specified path will then be imported as tasks in the current project.
- **Project level history of the Trained Models**. It is now possible to keep track of all previous training activities executed for a project. When pressing the History button from the Train page, users are presented with a list of all trainings triggered for the current project.
- **Easier page navigation**. Users can now right-click on the available links and select "Open in new tab" to open the link in a new tab without losing the current work context.
- **Optimized user editing UI**. All the checkboxes on the Users Edit page now have the same style. The "UserAdmins" group was renamed to "Admins" and the description of groups is more detailed and easier to understand. Also, a new error message is shown when an invalid email address is used.
- **Improved page navigation for Visual NER projects**. For Visual NER projects, users can jump to a specific page in any multi-page task instead of passing through all pages to reach a target section of a PDF document.
- **Visual configuration options for Visual NER project**. Users are now able to add custom labels and choices in the project configuration from the Visual tab for Visual NER projects as well as for the text projects.
- **Visual NER Models available on the Models Hub page**. Visual NER models can now be filtered, downloaded from the NLP Models Hub, and used for pre-annotating image-based documents.
- **Lower CPU and Memory resources allocated to the license server**. In this version, the resources allocated to the license server were decreased to CPU: 1000m (1 core) and Memory: 1GB.
- **Simplify Training and Pre-annotation configurations**. Now the user only need to adjust "Memory limit" and "CPU limit" in the Infrastructure page. "Spark Drive Memory" is calculated as 85% of Memory Limit where are "Spark Kryo Buff Max" and "Spark Driver Max Result Size" are constants with values "2000 MB" and "4096 MB" respectively.
- **Auto-close user settings**. The user settings menu is closed automatically when a user clicks on any other settings options.
- **Preserve task filters**. From version 4.3.0, all defined filters in the task page remain preserved when the user navigates back and forth between the labeling page and the task page.
- **Optimized Alert Messages**. All the alert notification shows clear errors, warnings, information, and success messages.
- **Zoom in/out features in Visual NER projects with Sticky Left Column view**. In various views of Visual NER, zoom-controlling features are now available by default.

</div><div class="prev_ver h3-box" markdown="1">

## Versions

</div>

{%- include docs-annotation-pagination.html -%}
