---
layout: docs
header: true
seotitle: NLP Server | John Snow Labs
title: Release Notes
permalink: /docs/en/nlp_server/nlp_server_versions/release_notes
key: docs-nlp-server
modify_date: "2022-06-17"
show_nav: true
sidebar:
  nav: nlp-server
---

<div class="h3-box" markdown="1">

## 0.7.1

| Fields       | Details    |
| ------------ | ---------- |
| Name         | NLP Server |
| Version      | `0.7.1`    |
| Type         | Patch      |
| Release Date | 2022-06-17 |

<br>

### Overview

We are excited to release NLP Server v0.7.1! We are committed to continuously improve the experience for our users and make our product reliable and easy to use.

This release focuses on solving a few bugs and improving the stability of the NLP Server.

<br>

### Key Information

1. For smooth and optimal performance, it is recommended to use an instance with 8 core CPU, and 32GB RAM specifications.
2. NLP Server is available on both [AWS](https://aws.amazon.com/marketplace/pp/prodview-4ohxjejvg7vwm) and [Azure](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/johnsnowlabsinc1646051154808.nlp_server) marketplaces.

<br>

### Bug Fixes

1. Issue when running NER ONTO spell.
2. Issue when running `dep` spell. Since the spell was broken it is temporarily blacklisted.
3. Document normalizer included the HTML, XML tags to the output even after normalization.
4. Issue when running language translation spells `<from_lang>.translate_to.<to_lang>`.
5. Upon cancelation of custom model uploading job exception was seen in the logs.
6. Some few UI related issues and abnormalities during operation.

</div><div class="prev_ver h3-box" markdown="1">

## Versions

</div>
{%- include docs-nlpserver-pagination.html -%}