---
layout: docs
comment: no
header: true
seotitle: Annotation Lab | John Snow Labs
title: Security and Privacy
permalink: /docs/en/alab/security
key: docs-training
modify_date: "2021-09-29"
use_language_switcher: "Python-Scala"
show_nav: true
sidebar:
    nav: annotation-lab
---


We understand and take the security issues as the highest priority. On every release, all our artifacts and images ran through a series of security testing - Static Code analysis, Pen Test, Images Vulnerabilities Test, AWS AMI Scan Test.  

Every identified critical issue is remediated, code gets refactored to pass our standard Static Code Analysis. 

## Role-based access 

Role-based access control is available for all Annotation Lab deployments. By default, all projects are private to the user who created them – the project owner. If necessary, project owners can add other users to the project and define their role(s) among annotator, reviewer, manager. The three roles supported by Annotation Lab offer different levels of task and feature visibility. Annotators can only see tasks assigned to them and their own completions. Reviewers can see the work of annotators who created completions for the tasks assigned to them. Annotators and reviewers do not have access to task import or annotation export nor to the Models Hub page. Managers have higher level of access. They can see all tasks content, can assign work to annotators and reviewers, can import tasks, export annotations, see completions created by team members or download models.  

When creating the annotation team, make sure the appropriate role is assigned to each team member according to the Need-To-Know Basis.  

Screen capture is not disabled, and given the high adoption of mobile technologies, team members can easily take pictures of the data. This is why, when dealing with sensitive documents, it is advisable to conduct periodical HIPPA/GDPR training with the annotation team to avoid data breaches.  

## Data sharing 

Annotation Lab runs locally - all computation and model training run inside the boundaries of your deployment environment.  The content related to any tasks within your projects is NOT SHARED with anyone.  

The Annotation Lab does not call home. Access to internet is used ONLY when downloading models from the NLP Models Hub. 

Document processing - OCR, pre-annotation, training, fine-tuning- runs entirely on your environment.  

Secure user access to Annotation Lab 

Access to Annotation Lab is restricted to users who are given access by an admin or project manager.  

Each user has an account; when created, passwords are enforced to best practice security policy.  

Annotation Lab keeps track of who has access to the defined projects and their actions regarding completions creation, cloning, submission, and starring. 

See [User Management Page](https://nlp.johnsnowlabs.com/docs/en/alab/user_management) for more details. 

## API access to Annotation Lab 

Access to Annotation Lab REST API requires an access token that is specific to a user account. To obtain your access token please follow the steps illustrated [here](https://nlp.johnsnowlabs.com/docs/en/alab/api#get-client-secret). 

## Complete project audit trail 

Annotation Lab keeps trail for all created completions. It is not possible for annotators or reviewers to delete any completions and only managers and project owners are able to remove tasks.  

 

## Application development cycle 

The  Annotation Lab development cycle currently includes static code analysis; everything is assembled as docker images whom are being scanned for vulnerabilities before being published. 

We are currently implementing web vulnerability scanning. 

 

 

