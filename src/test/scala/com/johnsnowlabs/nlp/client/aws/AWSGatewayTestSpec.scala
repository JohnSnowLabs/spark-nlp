package com.johnsnowlabs.nlp.client.aws

import com.amazonaws.auth.{AnonymousAWSCredentials, BasicSessionCredentials}
import com.johnsnowlabs.client.CredentialParams
import com.johnsnowlabs.client.aws.AWSTokenCredentials
import org.scalatest.FlatSpec

class AWSGatewayTestSpec extends FlatSpec {

  "AWSGatewayTestSpec" should "build Basic Credentials" in {
    val accessKeyId = "myAccessKeyId"
    val secretAccessKey = "mySecretAccessKey"
    val region = "myRegion"
    val credentialParams = CredentialParams(accessKeyId, secretAccessKey, "", "", region)

    val awsCredentials = new AWSTokenCredentials().buildCredentials(credentialParams)

    assert(awsCredentials.get.getAWSAccessKeyId == accessKeyId)
    assert(awsCredentials.get.getAWSSecretKey == secretAccessKey)
  }

  it should "build Token Credentials" in {
    val accessKeyId = "myAccessKeyId"
    val secretAccessKey = "mySecretAccessKey"
    val sessionToken = "mySessionToken"
    val region = "myRegion"
    val credentialParams = CredentialParams(accessKeyId, secretAccessKey, sessionToken, "", region)

    val awsCredentials = new AWSTokenCredentials().buildCredentials(credentialParams)

    val awsSessionCredentials: BasicSessionCredentials = awsCredentials.get.asInstanceOf[BasicSessionCredentials]

    assert(awsSessionCredentials.getAWSAccessKeyId == accessKeyId)
    assert(awsSessionCredentials.getAWSSecretKey == secretAccessKey)
    assert(awsSessionCredentials.getSessionToken == sessionToken)
  }

  it should "build provided credentials" in {
    val secretAccessKey = "mySecretAccessKey"
    val sessionToken = "mySessionToken"
    val region = "myRegion"
    val credentialParams = CredentialParams("", secretAccessKey, sessionToken, "", region)

    val awsCredentials = new AWSTokenCredentials().buildCredentials(credentialParams)

    assert(awsCredentials.isDefined)
  }

  it should "build anonymous credentials" in {
    val accessKeyId = "anonymous"
    val region = "myRegion"
    val credentialParams = CredentialParams(accessKeyId, "", "", "", region)

    val awsCredentials = new AWSTokenCredentials().buildCredentials(credentialParams)

    val awsSessionCredentials: AnonymousAWSCredentials = awsCredentials.get.asInstanceOf[AnonymousAWSCredentials]

    assert(awsSessionCredentials.getAWSAccessKeyId == null)
    assert(awsSessionCredentials.getAWSSecretKey == null)
  }

  it should "build Profile Credentials" in {
    val profile = "myProfile"
    val region = "myRegion"

    val credentialParams = CredentialParams("", "", "", profile, region)

    val awsCredentials = new AWSTokenCredentials().buildCredentials(credentialParams)

    assert(awsCredentials.isDefined)
  }

}
