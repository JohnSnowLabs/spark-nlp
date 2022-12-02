---
layout: docs
comment: no
header: true
seotitle: Annotation Lab | John Snow Labs
title: FAQ
permalink: /docs/en/alab/troubleshooting
key: docs-training
modify_date: "2022-11-20"
use_language_switcher: "Python-Scala"
show_nav: true
sidebar:
  nav: annotation-lab
---

<style>
  details {
    font-size: 16px;
    margin-bottom: 20px;
  }

  details > p:last-child {
    margin-bottom: 40px;
  }

  summary {
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 10px;
  }

  .anchor {
    font-size: 30px;
    font-weight: 600;
    margin-bottom: 20px;
  }

  .shell-output pre.highlight {
    background-color: #efefef !important;
    color: #4e4e4e;
  }

  .shell-output pre.highlight .nb {
    color: #00a7fa;
  }

  .shell-output code {
    font-family: monospace;
  }

  pre {
    max-height: 500px;
  }
</style>

Useful knowledge basebase for troubleshooting some of the common issues and tips for customizing the Annotation Lab set up and configurations.

<br />


<Element name="faq">

<details markdown="1">
<summary>1. How to deploy multiple preannotation/training servers in parallel?</summary>

By default the Annotation Lab installation is configured to use only one model server. If you want to allow the deployment of multiple model servers (e.g. up to 3), open the `annotationlab-upgrader.sh` script located under the `artifacts` folder of your Annotation Lab installation directory. Update the below configuration properties in the `annotaionlab-upgrader.sh` script for deploying upto 3 model servers.

```sh
--set airflow.model_server.count=3
--set model_server.count=3
```
Save the file and re-run this script for the changes to take effect.
</details>

<details markdown="1">
<summary>2. How can I access the API documentation?</summary>

API documentation is included in the Annotation Lab setup. So you will need to first set up Annotation Lab. Only _admin_ user can view the API documentation available under `Settings > API Integration`.

</details>

<details markdown="1">
<summary>3. Can I upload/download tasks/data using API?</summary>

Yes, it is possible to perform both the upload and download operations using API. There is import and export API for those operations. You can get more details about it from the API documentation.

</details>

<details markdown="1">
<summary>4. Can the user who created a project/task be assigned annotation/review tasks?</summary>

The project owner has by default all permissions (annotator, reviewer, manager). So we do not need to explicitly assign the annotator or reviewer role to the owner for the tasks.

</details>

<details markdown="1">
<summary>5. Can I download the swagger API documentation?</summary>

No. At present you can only access the API documentation directly from the API integration page under `Settings > API Integration`.

</details>

<details markdown="1">

<summary>6. How to uninstall Kubernetes during faulty install and re-install Annotation Lab?</summary>

If you have access to backend CLI then you can follow the steps below to fix faulty installation issue.

1. Go to /usr/local/bin

   ```sh
   cd /usr/local/bin
   ```

2. Run the uninstall script

   ```sh
   ./k3s-uninstall.sh
   ```

3. Re-run the installer script from the project folder

   ```sh
   ./k3s-installer.sh
   ```

4. Run the annotation lab installer

   ```sh
   ./annotationlab-installer.sh
   ```

This will take some time and produce the output below:

{:.shell-output}

```sh
NAME               STATUS   ROLES                  AGE     VERSION
ip-172-31-91-230   Ready    control-plane,master   3m38s   v1.22.4+k3s1
Image is up to date for sha256:18481c1d051558c1e2e3620ba4ddf15cf4734fe35dc45fbf8065752925753c9d
Image is up to date for sha256:a5b6ca180ebba94863ac9310ebcfacaaa64aca9efaa3b1f07ff4fad90ff76f68
Image is up to date for sha256:55208fe5388a7974bc4e3d63cfe20b2f097a79e99e9d10916752c3f8da560aa6
Image is up to date for sha256:a566a53e9ae7171faac1ce58db1d48cf029fbeb6cbf28cd53fd9651d5039429c
Image is up to date for sha256:09ad16bd0d3fb577cbfdbbdc754484f707b528997d64e431cba19ef7d97ed785
NAME: annotationlab
LAST DEPLOYED: Thu Sep 22 14:16:10 2022
NAMESPACE: default
STATUS: deployed
REVISION: 1
NOTES:
#############################################################################

Thank you for installing annotationlab. Please run the following commands to get the credentials.

export KEYCLOAK_CLIENT_SECRET_KEY=$(kubectl get secret annotationlab-secret --template={{.data.KEYCLOAK_CLIENT_SECRET_KEY}} | base64 --decode; echo)
export PG_PASSWORD=$(kubectl get secrets annotationlab-postgresql  -o yaml | grep '  postgresql-password:' | cut -d ' ' -f 4 | base64 -d; echo)
export PG_KEYCLOAK_PASSWORD=$(kubectl get secrets annotationlab-keyclo-postgres -o yaml | grep '  postgresql-password:' | cut -d ' ' -f 4 | base64 -d; echo)
export ADMIN_PASSWORD=$(kubectl get secret annotationlab-keyclo-admincreds --template={{.data.password}} | base64 --decode; echo)

#############################################################################
```
</details>
</Element>