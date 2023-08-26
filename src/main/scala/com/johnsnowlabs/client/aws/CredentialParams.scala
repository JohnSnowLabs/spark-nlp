package com.johnsnowlabs.client.aws

case class CredentialParams(
    accessKeyId: String,
    secretAccessKey: String,
    sessionToken: String,
    profile: String,
    region: String)
