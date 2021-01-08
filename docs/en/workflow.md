---
layout: docs
comment: no
header: true
title: Workflow Setup
permalink: /docs/en/workflow
key: docs-training
modify_date: "2020-11-18"
use_language_switcher: "Python-Scala"
---

When a team of people work together on a large annotation project, the tasks can be organized into a multi-step workflow for an easier management of the team collaboration. This is also necessary when the project has strict requirements on the labels: e.g. the same document must be labeled by multiple annotators; the annotations must be checked by a senior annotator.

In this situation, a workflow can be setup using the task tagging functionality provided by the Annotation Lab. Then can be used for splitting work across the team but also for differentiating between first-level annotators and second-level reviewers.  


<img class="image image--xl" src="/assets/images/annotation_lab/tags.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>


A full audit trail is kept on each action performed by all actors. Each saved entry is stored along with an authenticated user and a time stamp, and the user interface includes a visual comparison between versions.


To add a tag, select a task and press Tags > Add more. Tasks can be filtered by tags, making it easier to identify, for example, which documents are completed and which ones need to be reviewed.
 
<img class="image image--xl" src="/assets/images/annotation_lab/tags2.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>