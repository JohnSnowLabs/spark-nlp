---
layout: docs
comment: no
header: true
title: Import Documents 
permalink: /docs/en/import
key: docs-training
modify_date: "2020-11-18"
use_language_switcher: "Python-Scala"
---
Once a new project is created and its configuration is saved, the user is redirected to the **Import page**. Here the user has multiple options for importing tasks.

<img class="image image--xl" src="/assets/images/annotation_lab/image019.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

## Plain text file 
When you upload a text file, a task will be created for each line of that file. In other words the text is split by the new line character and a new task is created for each line. 


## Json file
For bulk importing a list of documents you can use the json import option. The expected format is illustrated in the image below. It consists of a list of dictionaries, each with 2 keys-values pairs  (“text” and “title”). 

```bash
[{  "text": "Task text content.", "title":"Task title"}]
```


## CSV, TSV file
When CSV / TSV formatted text file is used, column names are interpreted as task data keys:

```bash
Task text content, Task title
this is a first task, Colon Cancer.txt
this is a second task, Breast radiation therapy.txt
```