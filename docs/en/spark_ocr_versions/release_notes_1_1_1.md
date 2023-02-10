---
layout: docs
header: true
seotitle: Spark NLP
title: Spark NLP release notes 1.1.1
permalink: /docs/en/spark_ocr_versions/release_notes_1_1_1
key: docs-release-notes
modify_date: "2022-01-06"
show_nav: true
sidebar:
    nav: sparknlp
---

<div class="h3-box" markdown="1">

## 1.1.1

Release date: 06-03-2020

#### Overview

Integration with license server.

#### Enhancements

* Added license validation. License can be set in following waysq:
  - Environment variable. Set variable 'JSL_OCR_LICENSE'.
  - System property. Set property 'jsl.sparkocr.settings.license'.
  - Application.conf file. Set property 'jsl.sparkocr.settings.license'.
* Added auto renew license using jsl license server.


</div><div class="prev_ver h3-box" markdown="1">

## Versions

</div>
{%- include docs-sparckocr-pagination.html -%}