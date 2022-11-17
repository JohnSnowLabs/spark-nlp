---
layout: docs
comment: no
header: true
seotitle: Annotation Lab | John Snow Labs
title: Installation
permalink: /docs/en/alab/install
key: docs-training
modify_date: "2022-11-06"
use_language_switcher: "Python-Scala"
show_nav: true
sidebar:
  nav: annotation-lab
---

<style>
bl {
  font-weight: 400;
}
th {
  width: 200px;
  text-align: left;
  background-color: #f7f7f7;
  vertical-align: "top";
}
</style>

## Dedicated Server

Install Annotation Lab on a dedicated server to reduce the likelihood of conflicts or unexpected behavior.

### Fresh install

To install Annotation Lab run the following command:

```bash
wget https://setup.johnsnowlabs.com/annotationlab/install.sh -O - | sudo bash -s $VERSION
```

Replace `$VERSION` in the above one liners with the version you want to install.

For installing the latest available version of the Annotation Lab use:

```bash
wget https://setup.johnsnowlabs.com/annotationlab/install.sh -O - | sudo bash -s --
```

<br />

### Upgrade

To upgrade your Annotation Lab installation to a newer version, run the following command on a terminal:

```bash
wget https://setup.johnsnowlabs.com/annotationlab/upgrade.sh -O - | sudo bash -s $VERSION
```

Replace `$VERSION` in the above one liners with the version you want to upgrade to.

For upgrading to the latest version of the Annotation Lab, use:

```bash
wget https://setup.johnsnowlabs.com/annotationlab/upgrade.sh -O - | sudo bash -s --
```

> **NOTE:** The install/upgrade script displays the login credentials for the _admin_ user on the terminal.

After running the install/upgrade script, the Annotation Lab is available at http://INSTANCE_IP or https://INSTANCE_IP

<img class="image image--xl image__shadow" src="/assets/images/annotation_lab/4.1.0/loginScreenALAB.png" style="width:100%;"/>

We have an aesthetically pleasing Sign-In Page with a section highlighting the key features of Annotation Lab using animated GIFs.

## AWS Marketplace

Visit the [product page on AWS Marketplace](https://aws.amazon.com/marketplace/pp/prodview-nsww5rdpvou4w?sr=0-1&ref_=beagle&applicationId=AWSMPContessa) and follow the instructions on the video below to subscribe and deploy.

<div class="cell cell--12 cell--lg-6 cell--sm-12"><div class="video-item">{%- include extensions/youtube.html id='ebaewU4BcQA' -%}<div class="video-descr">Deploy Annotation Lab via AWS Marketplace</div></div></div>

## Azure Marketplace

Visit the [product page on Azure Marketplace](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/johnsnowlabsinc1646051154808.annotation_lab?tab=Overview) and follow the instructions on the video below to subscribe and deploy.

<div class="cell cell--12 cell--lg-6 cell--sm-12"><div class="video-item">{%- include extensions/youtube.html id='e6aB3z5tB0k' -%}<div class="video-descr">Deploy Annotation Lab via Azure Marketplace</div></div></div>


## EKS deployment

 1) Create NodeGroup for a given cluster


```console
eksctl create nodegroup --config-file eks-nodegroup.yaml

kind: ClusterConfig
apiVersion: eksctl.io/v1alpha5
metadata:
  name: <cluster-name>
  region: <region>
  version: "1.21"
availabilityZones:
  - <zone-1>
  - <zone-2>
vpc:
  id: "<vpc-id>"
  subnets:
    private:
      us-east-1d:
        id: "<subnet-id"
      us-east-1f:
        id: "<subent-id>"
  securityGroup: "<security-group>"
iam:
  withOIDC: true
managedNodeGroups:
  - name: alab-workers
    instanceType: m5.large
    desiredCapacity: 3
    VolumeSize: 50
    VolumeType: gp2
    privateNetworking: true
    ssh:
      publicKeyPath: <path/to/id_rsa_pub>

```
```console
eksctl utils associate-iam-oidc-provider --region=us-east-1 --cluster=<cluster-name> --approve
```

 2) Create an EFS as shared storage. EFS stands for Elastic File System and is a scalable storage solution that can be used for general purpose workloads. 


```console
curl -S https://raw.githubusercontent.com/kubernetes-sigs/aws-efs-csi-driver/v1.2.0/docs/iam-policy-example.json -o iam-policy.json
aws iam create-policy \ 
  --policy-name EFSCSIControllerIAMPolicy \ 
  --policy-document file://iam-policy.json
```
```console
eksctl create iamserviceaccount \ 
  --cluster=<cluster> \ 
  --region <AWS Region> \ 
  --namespace=kube-system \ 
  --name=efs-csi-controller-sa \ 
  --override-existing-serviceaccounts \ 
  --attach-policy-arn=arn:aws:iam::<AWS account ID>:policy/EFSCSIControllerIAMPolicy \ 
  --approve
```
```console
helm repo add aws-efs-csi-driver https://kubernetes-sigs.github.io/aws-efs-csi-driver
```
```console
helm repo update
```
```console
helm upgrade -i aws-efs-csi-driver aws-efs-csi-driver/aws-efs-csi-driver \ 
  --namespace kube-system \ 
  --set image.repository=602401143452.dkr.ecr.us-east-1.amazonaws.com/eks/aws-efs-csi-driver \ 
  --set controller.serviceAccount.create=false \ 
  --set controller.serviceAccount.name=efs-csi-controller-sa

```

 3) Create storageClass.yaml


```console
cat <<EOF > storageClass.yaml
kind: StorageClass
apiVersion: storage.k8s.io/v1
metadata:
  name: efs-sc
provisioner: efs.csi.aws.com
parameters:
  provisioningMode: efs-ap
  fileSystemId: <EFS file system ID>
  directoryPerms: "700"
EOF
```
```console
kubectl apply -f storageClass.yaml
```

```console
helm install annotationlab annotationlab-${ANNOTATIONLAB_VERSION}.tgz                                 \
    --kubeconfig /etc/rancher/k3s/k3s.yaml                                                            \
    --set image.tag=${ANNOTATIONLAB_VERSION}                                                          \
    --set model_server.count=1                                                                        \
    --set ingress.enabled=true                                                                        \
    --set networkPolicy.enabled=true                                                                  \
    --set networkPolicy.enabled=true --set extraNetworkPolicies='- namespaceSelector:
    matchLabels:
      kubernetes.io/metadata.name: kube-system
  podSelector:
    matchLabels:
      app.kubernetes.io/name: traefik
      app.kubernetes.io/instance: traefik'                                                            \
    --set keycloak.postgresql.networkPolicy.enabled=true                                              \
    --set persistence.storageClass=efs-sc                                                             \
    --set sharedData.storageClass=efs-sc                                                              \
    --set airflow.postgresql.networkPolicy.enabled=true                                               \
    --set postgresql.networkPolicy.enabled=true                                                       \
    --set airflow.networkPolicies.enabled=true                                                        \
    --set ingress.defaultBackend=true                                                                 \
    --set ingress.uploadLimitInMegabytes=16                                                           \
    --set 'ingress.hosts[0].host=domain.tld'                                                          \
    --set airflow.model_server.count=1                                                                \
    --set airflow.redis.password=$(bash -c "echo ${password_gen_string}")                             \
    --set configuration.FLASK_SECRET_KEY=$(bash -c "echo ${password_gen_string}")                     \
    --set configuration.KEYCLOAK_CLIENT_SECRET_KEY=$(bash -c "echo ${uuid_gen_string}")               \
    --set postgresql.postgresqlPassword=$(bash -c "echo ${password_gen_string}")                      \
    --set keycloak.postgresql.postgresqlPassword=$(bash -c "echo ${password_gen_string}")             \
    --set keycloak.secrets.admincreds.stringData.user=admin                                           \
    --set keycloak.secrets.admincreds.stringData.password=$(bash -c "echo ${password_gen_string}")

```


 4) Install ingress Controller


```
helm repo add nginx-stable https://helm.nginx.com/stable
helm repo update
helm install my-release nginx-stable/nginx-ingress
```


 5) Apply ingress.yaml 


```console
cat <<EOF > ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  annotations:
    kubernetes.io/ingress.class: nginx
    meta.helm.sh/release-name: annotationlab
    meta.helm.sh/release-namespace: default
  name: annotationlab
spec:
  defaultBackend:
    service:
      name: annotationlab
      port:
        name: http
  rules:
  - host: domain.tld
    http:
      paths:
      - backend:
          service:
              name: annotationlab
              port:
                name: http
        path: /
        pathType: ImplementationSpecific
      - backend:
          service:
              name: annotationlab-keyclo-http
              port:
                name: http
        path: /auth
        pathType: ImplementationSpecific
EOF  
```  
```console
kubectl apply -f ingress.yaml
```


## AirGap Environment

### Get Artifact

Run the following command on a terminal to fetch the compressed artifact (_tarball_) of the Annotation Lab.

```bash
wget https://s3.amazonaws.com/auxdata.johnsnowlabs.com/annotationlab/annotationlab-$VERSION.tar.gz
```

Extract the tarball and the change directory to the extracted folder (_artifacts_):

```bash
tar -xzf annotationlab-$VERSION.tar.gz
cd artifacts
```

Replace `$VERSION` with the version you want to download and install.

<br />

### Fresh Install

Run the installer script `annotationlab-installer.sh` with `sudo` privileges.

```bash
$ sudo su
$ ./annotationlab-installer.sh
```

<br />

### Upgrade

Run the upgrade script `annotationlab-updater.sh` with `sudo` privileges.

```bash
$ sudo su
$ ./annotationlab-updater.sh
```

<br />

### Work over proxy

**Custom CA certificate**

You can provide a custom CA certificate chain to be included into the deployment. To do it add `--set-file custom_cacert=./cachain.pem` options to `helm install/upgrade` command inside `annotationlab-installer.sh` and `annotationlab-updater.sh` files.

_cachain.pem_ must include a certificate in the following format:

```bash
-----BEGIN CERTIFICATE-----
....
-----END CERTIFICATE-----
```

<br />

**Proxy env variables**

You can provide a proxy to use for external communications. To do that add

    `--set proxy.http=[protocol://]<host>[:port]`,
    `--set proxy.https=[protocol://]<host>[:port]`,
    `--set proxy.no=<comma-separated list of hosts/domains>`

commands inside `annotationlab-installer.sh` and `annotationlab-updater.sh` files.

## Recommended Configurations

<table>
  <tr>
    <th>System requirements</th>
    <td>You can install Annotation Lab on a Ubuntu 20+ machine.</td>
  </tr>
  <tr>
    <th>Port requirements</th>
    <td>Annotation Lab expects ports <bl>443</bl> and <bl>80</bl> to be open by default.</td>
  </tr>
  <tr>
    <th>Server requirements</th>
    <td>The minimal required configuration is <bl>32GB RAM, 8 Core CPU, 512 SSD</bl>.<br /><br />

    The ideal configuration in case model training and preannotations are required on a large number of tasks is <bl>64 GiB, 16 Core CPU, 2TB HDD, 512 SSD</bl>.
    </td>

  </tr>
  <tr>
    <th>Web browser support</th>
    <td>Annotation Lab is tested with the latest version of Google Chrome and is expected to work in the latest versions of:
      <ul>
      <li>Google Chrome</li>
      <li>Apple Safari</li>
      <li>Mozilla Firefox</li>
      </ul>
    </td>
  </tr>
</table>
