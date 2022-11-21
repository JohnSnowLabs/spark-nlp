---
layout: docs
comment: no
header: true
seotitle: Annotation Lab | John Snow Labs
title: Analytics Permission
permalink: /docs/en/alab/analytics_permission
key: docs-training
modify_date: "2022-10-14"
use_language_switcher: "Python-Scala"
show_nav: true
sidebar:
  nav: annotation-lab
---

By default, dashboards in the Analytics page is disabled for a project. Users can request the _admin_ to enable the Analytics page. The request is then listed on the Analytics Request page under the Settings menu. This page is only accessible to the _admin_ user. After the _admin_ user approves the request, the user can access the various dashboards in the Analytics page.

## Analytics Requests

The Analytics Requests page lists all the pending requests for the Analytics page from one or more users. The _admin_ user can grant or deny the permission to the requests as needed. It is accessible from `Settings > Analytics Requests`. Each request contains information such as the name of project for which the analytics request was made, the user who initiated the request, and the date when the request was made.

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/analytics_requests.png" style="width:100%;"/>

**Granting a request**

All the requests granted by the _admin_ user is listed under this tab. The table shows information about the granted requests, like the name of the project for which the analytics request was made, the user who initiated the request, the user who granted the request, the date when the request was granted, the latest date when the analytics were updated. The _admin_ user can also revoke an already granted request from this list.

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/ar_granted.png" style="width:100%;"/>

**Denying/Revoking a request**

All the requests denied or revoked by the _admin_ user is listed under this tab. The table shows information about the denied/revoked requests, like the name of the project for which the analytics request was made, the user who initiated the request, the user who denied/revoked the request, the date when the request was denied/revoked, the latest date when the analytics were updated.

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/ar_revoked.png" style="width:100%;"/>
