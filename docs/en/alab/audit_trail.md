---
layout: docs
comment: no
header: true
seotitle: Annotation Lab | John Snow Labs
title: Audit Trail
permalink: /docs/en/alab/audit_trail
key: docs-training
modify_date: "2022-10-31"
use_language_switcher: "Python-Scala"
show_nav: true
sidebar:
    nav: annotation-lab
---

<style>
bl {
  font-weight: 400;
}

es {
  font-weight: 400;
  font-style: italic;
}
</style>

Annotation Lab is designed to handle <bl>Personal Identifying Information (PII)</bl> and <bl>Protected Health Information (PHI)</bl>. It keeps a full audit trail for all created completions, where each entry is stored with an authenticated user and a timestamp. It is not possible for <es>Annotators</es> or <es>Reviewers</es> to delete any completions, and only <es>Managers</es> and <es>Project Owners</es> can remove tasks.

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/audit_trail.png" style="width:100%;" />
