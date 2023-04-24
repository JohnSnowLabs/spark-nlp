---
layout: docs
header: true
seotitle: Annotation Lab | John Snow Labs
title: Annotation Lab Release Notes 4.8.4
permalink: /docs/en/alab/annotation_labs_releases/release_notes_4_8_4
key: docs-licensed-release-notes
modify_date: 2023-04-14
show_nav: true
sidebar:
  nav: annotation-lab
---

<div class="h3-box" markdown="1">

## 4.8.4

Release date: **13-04-2023**

NLP Lab v4.8.4, which includes stabilization and bugfixes. The following are some of the key updates included in this release:

- Improvements in keycloak resources api calls with proper error handling
- Get_server error is seen in annotationlab pod when user navigate to clusters page
- When the user selects a new label, the chunk that was previously unselected becomes labeled
- The user is not able to select multi-line text in the visual NER task using the post-annotation gesture
- For a multi-paged task, user is not able to annotate texts with a label when text is selected first.
- Training fails when the training type is Assertion
- Deployment crashes for Prompt without false_prompts parameter



</div><div class="prev_ver h3-box" markdown="1">

## Versions

</div>

{%- include docs-annotation-pagination.html -%}