package com.johnsnowlabs.client.aws

import com.amazonaws.auth.AWSCredentials
import com.johnsnowlabs.client.CredentialParams

trait Credentials {

  val next: Option[Credentials] = None

  def buildCredentials(credentialParams: CredentialParams): Option[AWSCredentials]

}
