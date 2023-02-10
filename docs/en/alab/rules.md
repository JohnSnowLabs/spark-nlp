---
layout: docs
comment: no
header: true
seotitle: NLP Lab | John Snow Labs
title: Rules
permalink: /docs/en/alab/rules
key: docs-training
modify_date: "2022-12-10"
use_language_switcher: "Python-Scala"
show_nav: true
sidebar:
  nav: annotation-lab
---

<style>
es {
    font-weight:400;
    font-style: italic;
}
</style>

Rule based annotation is supported by Healthcare NLP, Finance NLP, and Legal NLP via the <es>ContextualParser</es> Annotator.
Annotation Lab supports creating and using rules in a <es>NER</es> project using any one of these libraries with the presence of valid license.

Users in the <es>Admins</es> group can see and edit the available rules on the Rules page under the Models Hub menu. Users can create new rules using the _+ Add Rules_ button. Users can also import and export the rules.

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/rules.png" style="width:100%;"/>

There are two types of rules supported:

- **Regex Based:** Users can define a regex that will be used to label all possible hit chunks and label them as the target entity. For example, for labeling height entity the following regex can be used `[0-7]'((0?[0-9])|(1(0|1)))''`. All hits found in the task's text content that match the regex are pre-annotated as _height_.

- **Dictionary-Based:** Users can define and upload a CSV dictionary of keywords that cover the list of chunks that should be annotated as a target entity. For example, for the label _female_, all occurrences of strings _woman_, _lady_, and _girl_ within the text content of a given task will be pre-annotated as _female_.

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/types_of_rules.png" style="width:100%;"/>

After adding a rule, the <es>Project Owner</es> or <es>Manager</es> can add the rule to the configuration of the project where they want to use it. This can be done from the <es>Rules</es> screen of the <es>Project Configuration</es> step on the <es>Project Setup</es> page. A valid Healthcare, Finance or Legal NLP license is required to deploy rules as a pre-annotation server after completing the project configuration step.

The user is notified every time a rule in use is edited with the message _"Redeploy preannotation server to apply these changes_" on the <es>Edit Rule</es> form.

<img class="image image__shadow image__align--center" src="/assets/images/annotation_lab/4.1.0/redeploy_rules.png" style="width:70%;"/>

<br>

### Import and Export Rules

Annotation Lab allows importing and exporting Rules from the Rules page.

**Import Rules**

Users can import rules from the Rules page. The rules can be both _dictionary_ based or _regex_ based. The rules can be imported in the following formats:

1. JSON file or content.
2. Zip archive of JSON file/s.

![RulesImport](/assets/images/annotation_lab/4.1.0/import_rules.gif)

**Export Rules**

To export any rule, the user need to select the available rules and click on _Export Rules_ button. Rules are then downloaded as a zip file. The zip file contains the JSON file for each rule. These exported rules can again be imported to Annotation Lab.

![RulesExport](/assets/images/annotation_lab/4.1.0/export_rules.gif)

The following blog posts explain how to create and use rules for jump starting your annotation projects:

- [Using Rules to Jump Start Text Annotation Projects](https://medium.com/annotation-lab/using-rules-to-jump-start-text-annotation-projects-1-3-john-snow-labs-8277a9c7fbcb)
- [Using Rules and Pretrained Models in Text Annotation Projects](https://medium.com/annotation-lab/using-rules-and-pretrained-models-in-text-annotation-projects-2-3-1c6708d6c42d)
- [Training and tuning models based on Rule-based annotation of text documents](https://medium.com/annotation-lab/training-and-tuning-models-based-on-rule-based-annotation-of-text-documents-3-3-3bc89214d7b4)
