---
layout: docs
header: true
seotitle: Install Software - John Snow Labs
pagetitle: Install Software - John Snow Labs
title: Installation
permalink: /docs/en/nlp_server/installation
key: docs-nlp-server
modify_date: "2021-09-22"
show_nav: true
sidebar:
  nav: nlp-server
---

## Deploy using Docker

For deploying NLP Server on your instance run the following command.

```shell

docker run --pull=always -p 5000:5000 johnsnowlabs/nlp-server:latest

```

This will check if the latest docker image is available on your local machine and if not it will automatically download and run it.

If you want to keep downloaded models between restarts of the docker image, you can mount a volume.

```shell
mkdir /var/cache_pretrained
chown 1000:1000 /var/cache_pretrained
docker run --pull=always -v /var/cache_pretrained:/home/johnsnowlabs/cache_pretrained -p 5000:5000 johnsnowlabs/nlp-server:latest
```

## Deploy using AWS Marketplace

NLP Server on AWS Marketplace provides one of the fastest and easiest ways to get up and running on Amazon Web Services (AWS). NLP Server is available through AWS Marketplace free of charge. However, to use licensed spells in NLP Server, you need to buy our license from [here](https://www.johnsnowlabs.com/install/).

You can get NLP Server on AWS Marketplace from [this URL](https://aws.amazon.com/marketplace/pp/prodview-4ohxjejvg7vwm?sr=0-2&ref_=beagle&applicationId=AWSMPContessa).

Follow the seven steps instructions or the video tutorial given below to learn how to deploy NLP Server using AWS Marketplace. Make sure you have a valid AWS account and log in to the AWS Marketplace using your credentials.

<div class="cell cell--12 cell--lg-6 cell--sm-12"><div class="video-item">{%- include extensions/youtube.html id='3_T0P397i-k' -%}<div class="video-descr">Deploy NLP Server via AWS Marketplace</div></div></div>

1.Click on `Continue to subscribe` button for creating a subscription to the NLP Server product. The software is free of charge.

<img class="image image--xl" src="/assets/images/nlp_server/AWS_s1.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

2.Read the subscription EULA and click on `Accept terms` button if you want to continue.

<img class="image image--xl" src="/assets/images/nlp_server/AWS_s2.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

3.In a couple of seconds the subscription becomes active.

<img class="image image--xl" src="/assets/images/nlp_server/AWS_s3.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

Once it is active you see this screen. 

<img class="image image--xl" src="/assets/images/nlp_server/AWS_s4.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

4.Go to AWS Marketplace > Manage subscriptions and click on the `Launch new instance` button corresponding to the NLP Server subscription.
This will redirect you to the following screen. Click on `Continue to launch through EC2` button.

<img class="image image--xl" src="/assets/images/nlp_server/EC2_0.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

5.From the available options select the instance type you want to use for the deployment. Then click on `Review and Lauch` button.
<img class="image image--xl" src="/assets/images/nlp_server/EC2_s2.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

6.Select an existing key pair or create a new one. This ensures a secured connection to the instance. If you create a new key make sure that you download and safely store it for future usage. Click on the `Launch` button.

<img class="image image--xl" src="/assets/images/nlp_server/EC2_s3.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

7.While the instance is starting you will see this screen.

<img class="image image--xl" src="/assets/images/nlp_server/EC2_s4.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

Then the instance will appear on your EC2 Instances list.

<img class="image image--xl" src="/assets/images/nlp_server/EC2_s5.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

The NLP Server can now be accessed via a web browser at http://PUBLIC_EC2_IP .

<img class="image image--xl" src="/assets/images/nlp_server/EC2_s6.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

API documentation is also available at http://PUBLIC_EC2_IP/docs

<img class="image image--xl" src="/assets/images/nlp_server/EC2_s7.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

## Deploy using Azure Marketplace

NLP Server on Azure Marketplace provides one of the fastest and easiest ways to get up and running on Microsoft Azure. NLP Server is available through Azure Marketplace free of charge. However, to use licensed spells in NLP Server, you need to buy our license from [here](https://www.johnsnowlabs.com/install/).

You can get NLP Server on Azure Marketplace from [this URL](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/johnsnowlabsinc1646051154808.nlp_server).

Follow the video tutorial given below to learn how to deploy NLP Server using Azure Marketplace.

<div class="cell cell--12 cell--lg-6 cell--sm-12"><div class="video-item">{%- include extensions/youtube.html id='isxffn4Tcds' -%}<div class="video-descr">Deploy NLP Server using Azure Marketplace</div></div></div>
