---
layout: docs
header: true
seotitle: Annotation Lab | John Snow Labs
title: Annotation Lab Release Notes 4.5.0
permalink: /docs/en/alab/annotation_labs_releases/release_notes_4_5_0
key: docs-licensed-release-notes
modify_date: 2023-01-01
show_nav: true
sidebar:
  nav: annotation-lab
---

<div class="h3-box" markdown="1">

## 4.5.0

Release date: **01-01-2023**

Over the last year, Annotation Lab has grown to be much more than a document annotation tool. It became a full-fledged AI system, capable of testing pre-trained models and rules, applying them to new datasets, training, and tuning models, and exporting them to be deployed in production. All those features together with the new Playground concept presented in the current release notes contributed to the transformation of the Annotation Lab into the **NLP Lab**.
A new Playground feature is released as part of the NLP Lab’s Hub that allows users to quickly test any model and/or rule on a snippet of text without the need to create a project and import tasks. 
NLP Lab also supports the training of Legal and Finance models and Model evaluation for classification projects. 
As always the release includes some stabilization and bug fixes for issues reported by our user community. Below are the details of what has been included in this release.

## NLP Lab's Playground 
NLP Lab introduces the Playground feature where users can directly deploy and test models and/or rules. In previous versions, the pre-annotation servers could only be deployed from within a given project. With the addition of the Playground, models can easily be deployed and tested on a sample text without going through the project setup wizard. Any model or rule can now be selected and deployed for testing by clicking on the "Open in Playground" button.

![Playground deployment](https://user-images.githubusercontent.com/33893292/209965776-1c0a6b07-5526-496e-97f8-4b7a2cf2a6d1.gif)

Rules are deployable in the Playground from the rules page. When a particular rule is deployed in the playground, the user can also change the parameters of the rules on the right side of the page. After saving the changes users need to click on the "Deploy" button to refresh the results of the pre-annotation on the provided text.

![ruledeploy](https://user-images.githubusercontent.com/33893292/209978724-355fd3a2-aab4-4c38-869d-4f3432c657d3.gif)

Deployment of models and rules is supported by floating and air-gapped licenses. Healthcare, Legal, and Finance models require a license with their respective scopes to be deployed in Playground. Unlike pre-annotation servers, only one playground can be deployed at any given time.

## Export trained models to the S3 bucket

With this release, users can easily export trained models to a given s3 bucket. This feature is available on the Available Models page under the Hub tab. Users need to enter the S3 bucket path, S3 access key, and S3 secret key to upload the model to the S3 bucket.

 ![ezgif com-gif-maker (1)](https://user-images.githubusercontent.com/73094423/210034029-4b1816bd-78dd-4551-ae6e-0303b58be8f0.gif)


## Support Training of Finance and Legal Models
With this release, users can perform training of Legal and Finance models depending on the available license(s). When training a new model in the NLP Lab, users have the option to select what library to use. Two options were available up until now: Open source and Healthcare. This release adds two new options: Legal and Finance. This helps differentiate the library used for training the models. The new options are only available when at least one valid license with the corresponding scope is added to the License page.

![Screenshot 2022-12-29 at 10 47 38 PM](https://user-images.githubusercontent.com/33893292/209985787-e03f9125-c716-4897-a1c9-47f91191d9bc.png)

## Improvements
### Keyword-based Search at task level

Finding tokens on the Visual NER project was restricted to only one page, and searching for keywords from the labeling page on a text-based project was not available.

NLP Lab supports task-level keyword-based searches. The keyword-based search feature will work for text and Visual NER projects alike.

- The search will work on all paginated pages.
- It is also possible to navigate between search results, even if that result is located on another page.

**Important**

Previously this feature was implemented with the help of <search> tag in the Visual NER project configurations. With the implementation of search at task level, the previous search tag should be removed from existing visual NER projects.

 _**Config to be removed from all existing Visual NER project:**_

```
<Search name="search" toName="image" placeholder="Search"/>
```

![text-search](https://user-images.githubusercontent.com/45035063/209945478-541a9d48-33b4-418c-8ea1-08d87b249e85.gif)

![vOCR-search](https://user-images.githubusercontent.com/45035063/209945481-b4b3127b-ce2a-49af-a3cd-f35dbe7771a9.gif)


### Chunk-based Search in Visual NER tasks

In previous versions, users could only run token-based searches at page level. The search feature did not support searching a collection of tokens as a single chunk. With this release, users can find a chunk of tokens in the Visual NER task.

![chunk-search](https://user-images.githubusercontent.com/45035063/209945712-c175f74a-b232-47f9-b807-4966ff54d976.gif)

### Model Evaluation for Classification Projects

![Screen Shot 2022-12-14 at 6 01 34 PM](https://user-images.githubusercontent.com/45035063/210033230-1044c26e-8925-4077-8113-dccca09ed736.png)

Up until now, the Annotation Lab only supported test and model evaluation for the NER-based projects. From this version on, NLP Lab supports test and model evaluation for Classification project as well. Evaluation results can now be downloaded if needed. 

### Hide and Unhide Regions in NER project 

In this version, we support the hide/show annotated token regions feature in the text-based project in the same way as it was available in the Visual NER project.

![hide-unhide-region](https://user-images.githubusercontent.com/45035063/209945808-f99043cd-a9fc-41a8-a85f-41bebea73512.gif)

### Ground Truth can only be set/unset by the owner of the completion

With this version, we have improved the feature to set/unset ground truth for a completion submitted by an annotator. Now, for the Manager/Project Owner/Reviewer, the button to set/unset ground truth is disabled. The ground truth can only be updated by the annotator who submitted the completion or is unset when a submitted completion is rejected by a reviewer. 

![Screen Recording 2022-12-29 at 3 23 25 PM](https://user-images.githubusercontent.com/17021686/209933021-33b4c98a-fbbc-46e4-aae1-b858e8c73fe6.gif)


### Finite Zoom Out Level in Visual NER tasks

Previously, users could zoom in and zoom out again on images while working with the Visual NER project, but the user could not get what the last stage of zoom-out was. Now, when the user zooms out of the image if it is the last phase then the zoom-out button will automatically be disabled so the user knows where to stop zooming out next.

![image](https://user-images.githubusercontent.com/73094423/210032879-36b86d98-5f13-40ea-836b-f2b61d63f4bf.png)

### Taxonomy Location Customizable from the Project Configuration

There are many different views available for each project template. This diversity can be confusing for users. For eliminating this complexity, the View tab was removed from the project configuration page and replaced by an “orientation” option that can be directly applied to the project configuration. Orientation will decide, where the taxonomy (labels, choices, text, images, etc.) will be located on the labeling screen i.e. placed at the top, bottom or next to the annotation screen.

<img width="1140" alt="Screenshot 2022-12-30 at 10 57 13 AM" src="https://user-images.githubusercontent.com/33893292/210036948-70185f91-e0c9-4dbc-af75-622f2185a08f.png">

### Pre-annotation CPU requirement message in Visual NER projects

By default, the pre-annotation server uses 2 CPUs. For Visual NER pre-annotation, it is likely that 2 CPUs are not enough. Now a friendly message is shown during the deployment of Visual NER pre-annotation if the CPU count is less than or equal to 2.


## Bug Fixes

- **Expanding the text on the Labelling page visually does not expand the labeling area**

Previously, expanding the text area on the labeling page did not make any changes in the text expansion. This issue has been fixed. Now, expanding the text will change the text area to full-screen mode.

- **Revoking granted analytics request do not update the revoked section** 

Earlier, when an analytics request was revoked, the corresponding entry was not shown in the revoked section. We have fixed this issue. With NLP Lab 4.5.0, the revoked entries are available in the revoked section. Also, when an analytics request is revoked, in the revoked section, two new actions, Accept and Delete, are available.

- **Show Confidence score in Regions option is not working properly for non-Visual NER tasks**

For none Visual NER tasks, enabling/disabling "Show Confidence score in Regions" from Layout did not change the UI. The changes only appear when the page was reloaded or when the Versions tab was clicked. This issue has been fixed in this version.

- **Username validation is missing when creating a new user**

With this version, the issue related to the missing validation of the username when creating a new user has been fixed.

- **Issues with role selection on Teams page**

When a user was added to the project team as a new team member, the recently added user name was still visible in the search bar. This issue has been fixed.

- **Clicking on the eye icon to hide a labeled region removes the region from the Annotations widget**

Previously, when a user clicked on the eye icon to hide a label, the labeled region was removed from the Annotations widget. Furthermore, the color of the label was also changed in the panel. This issue has been fixed.

- **Deployed legal and finance models servers are not associated with their respective licenses**

In the previous version, when a Legal and Finance model server was deployed, the respective licenses were not associated with their deployed server. The availability of the Legal and Finance license was checked when the models were deployed. Version 4.5.0 fixes this bug.

- **Model Evaluation cannot be triggered using an air-gapped healthcare license with scope training/inference**

The issue of triggering Model Evaluation using an air-gapped healthcare license with the training/inference scope has been fixed. 

- **When user enabled "Allow user for custom selection of regions", token values are missing in JSON export**

Earlier, when the user annotates tokens while enabling "Allow user for custom selection of regions" and exports the completion. The token values were missing from the JSON export. In this version, the issue is fixed, and all the token fields and values are available in the JSON

- **Pre-annotation server with pending status is not removed when the user deletes the server from the cluster page**

Deleting the pre-annotation server with status pending from the cluster page did not delete the pod from Kubernetes and created multiple pre-annotation pods. This issue has been fixed. 

- **Project export with space in the name is allowed to be imported**

In the earlier version, the users could import previously exported projects with space in the project's name. Though the project was listed on the projects page, the project could not be deleted. Also, the user was unable to perform any operations on the project.

- **The “Only Assigned” checkbox overlaps the review dialog box**

The overlap between the “Only Assigned” checkbox and the review dialog box was fixed. 

- **Open-source Models cannot be downloaded in the NLP Lab without a license**

Previously open-source models could not be downloaded from the NLP models hub when there was no license uploaded. This issue has been fixed. Now all open-source licenses are downloadable without any issue.

</div><div class="prev_ver h3-box" markdown="1">

## Versions

</div>

{%- include docs-annotation-pagination.html -%}