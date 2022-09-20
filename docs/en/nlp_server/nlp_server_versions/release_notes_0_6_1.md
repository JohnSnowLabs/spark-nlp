---
layout: docs
header: true
seotitle: NLP Server | John Snow Labs
title: NLP Server release notes 0.6.1
permalink: /docs/en/nlp_server/nlp_server_versions/release_notes_0_6_1
key: docs-release-notes
modify_date: "2022-02-09"
show_nav: true
sidebar:
    nav: sparknlp
---

## NLP Server 0.6.1

| Fields       | Details    |
| ------------ | ---------- |
| Name         | NLP Server |
| Version      | `0.6.1`    |
| Type         | Patch      |
| Release Date | 2022-05-06 |

<br>

### Overview

We are excited to release NLP Server v0.6.1! We are continually committed towards improving the experience for our users and making our product reliable and easy to use.

This release focuses on improving the stability of the NLP Server and cleaning up some annoying bugs. To enhance the user experience, the product now provides interactive and informative responses to the users.

The improvements and bug fixes are mentioned in their respective sections below.

<br>

### Key Information

1. For smooth and optimal performance, it is recommended to use an instance with 8 core CPU, and 32GB RAM specifications.
2. NLP Server is available on both [AWS](https://aws.amazon.com/marketplace/pp/prodview-4ohxjejvg7vwm) and [Azure](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/johnsnowlabsinc1646051154808.nlp_server) marketplace.

<br>

### Improvements

1. Support for new models for Lemmatizers, Parts of Speech Taggers, and Word2Vec Embeddings for over 66 languages, with 20 languages being covered for the first time by NLP Server, including ancient and exotic languages like Ancient Greek, Old Russian, Old French and much more.

<br>

### Bug Fixes

1. The prediction job runs in an infinite loop when using certain spells. Now after 3 retries it aborts the process and informs users appropriately.
2. Issue when running lang spell for language classification.
3. The prediction job runs in an infinite loop when incorrect data format is selected for a given input data.
4. The API request for processing spell didn’t work when format parameter was not provided. Now it uses a default value in such case.
5. Users were unable to login to their MYJSL account from NLP Server.
6. Proper response when there is issue in internet connectivity when running spell.


<div class="prev_ver h3-box" markdown="1">

## Versions

</div>

<ul class="pagination">
    <li>
        <a href="release_notes_0_6_0">Version 0.6.0</a>
    </li>
    <li>
        <strong>Version 0.6.1</strong>
    </li>
    <li>
        <a href="release_notes_0_7_0">Version 0.7.0</a>
    </li>
</ul>

<ul class="pagination owl-carousel pagination_big">
  <li><a href="release_notes_0_7_1">0.7.1</a></li>
  <li><a href="release_notes_0_7_0">0.7.0</a></li>
  <li class="active"><a href="release_notes_0_6_1">0.6.1</a></li>
  <li><a href="release_notes_0_6_0">0.6.0</a></li>
  <li><a href="release_notes_0_5_0">0.5.0</a></li>
  <li><a href="release_notes_0_4_0">0.4.0</a></li>
</ul>