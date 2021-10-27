---
layout: docs
comment: no
header: true
seotitle: Visual NER | John Snow Labs
title: Visual NER
permalink: /docs/en/alab/visual_ner
key: docs-training
modify_date: "2021-10-27"
use_language_switcher: "Python-Scala"
show_nav: true
sidebar:
    nav: annotation-lab
---



## API access to Annotation Lab 

Access to Annotation Lab REST API requires an access token that is specific to a user account. To obtain your access token please follow the steps illustrated [here](https://nlp.johnsnowlabs.com/docs/en/alab/api#get-client-secret). 

## Complete project audit trail 

Annotation Lab keeps trail for all created completions. It is not possible for annotators or reviewers to delete any completions and only managers and project owners are able to remove tasks.  

 

## Application development cycle 

The  Annotation Lab development cycle currently includes static code analysis; everything is assembled as docker images whom are being scanned for vulnerabilities before being published. 

We are currently implementing web vulnerability scanning. 

 

 

