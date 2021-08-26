package com.johnsnowlabs.client.aws

import com.amazonaws.auth.{AWSCredentials, AnonymousAWSCredentials}
import com.johnsnowlabs.client.CredentialParams

class AWSAnonymousCredentials extends Credentials {

  override def buildCredentials(credentialParams: CredentialParams): Option[AWSCredentials] = {
    if (credentialParams.region != "") {
      logger.info("Connecting to AWS with Anonymous AWS Credentials...")
      return Some(new AnonymousAWSCredentials())
    }
    None
  }

}
