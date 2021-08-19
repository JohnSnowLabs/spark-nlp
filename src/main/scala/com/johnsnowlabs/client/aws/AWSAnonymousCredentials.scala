package com.johnsnowlabs.client.aws

import com.amazonaws.auth.{AWSCredentials, AnonymousAWSCredentials}
import com.johnsnowlabs.client.CredentialParams

class AWSAnonymousCredentials extends Credentials {

  override def buildCredentials(credentialParams: CredentialParams): Option[AWSCredentials] = {
    if (credentialParams.region != "") {
      return Some(new AnonymousAWSCredentials())
    }
    None
  }

}
