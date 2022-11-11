---
layout: docs
comment: no
header: true
seotitle: Annotation Lab | John Snow Labs
title: Workflows
permalink: /docs/en/alab/workflow
key: docs-training
modify_date: "2022-11-02"
use_language_switcher: "Python-Scala"
show_nav: true
sidebar:
  nav: annotation-lab
---

When a team of people collaborate on a large annotation project, the work can be organized into multi-step workflows for an easier management of each team member's responsabilities. This is also necessary when the project has strict requirements such as the same document must be labeled by multiple annotators; the annotations must be checked by a senior annotator.

The default workflow supported by the Annotation Lab involves task assignment to one or multiple Annotators and to maximum one Reviewer. In the majority of projects having one annotator working on a task and then one reviewer checking the work done by the annotator is sufficient. 

> **NOTE:** In NER projects, we recommend that in the early stages of a project, a batch of 50 - 100 content rich tasks should be assigned to all annotators for checking the Inter Annotator Agreement (IAA). This is a best practice to follow in order to quickly identify the difference in annotations as a complementary way to ensure high agreement and completeness across team. 

> **NOTE:** When multiple annotators are assigned to a task, multiple ground truth completions will be created for that task. The way Annotation Lab prioritises the ground truth completion used for model training and CONLL export is via the priority assigned for each user in the Team (see [Project Configuration](/docs/en/alab/project_creation#adding-team-members)). 


When more complex workflows need to be implemented, this is possible using the task tagging functionality provided by the Annotation Lab. Tags can be used for splitting work across the team but also for differentiating between first-level annotators and second-level reviewers.

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/tags.png" style="width:100%;"/>


To add a tag, select a task and press _Tags > Add More_. Tasks can be filtered by tags, making it easier to identify, for example, which documents are completed and which ones need to be reviewed.

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/tag_assign.png" style="width:100%;"/>
