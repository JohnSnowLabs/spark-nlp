/*
 * Copyright 2017-2022 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.nlp.client.aws

import com.amazonaws.auth.{AnonymousAWSCredentials, BasicSessionCredentials}
import com.johnsnowlabs.client.aws.{AWSTokenCredentials, CredentialParams}
import com.johnsnowlabs.tags.SlowTest
import org.scalatest.flatspec.AnyFlatSpec

class AWSGatewayTestSpec extends AnyFlatSpec {

  "AWSGatewayTestSpec" should "build Basic Credentials" taggedAs SlowTest in {
    val accessKeyId = "myAccessKeyId"
    val secretAccessKey = "mySecretAccessKey"
    val region = "myRegion"
    val credentialParams = CredentialParams(accessKeyId, secretAccessKey, "", "", region)

    val awsCredentials = new AWSTokenCredentials().buildCredentials(credentialParams)

    assert(awsCredentials.get.getAWSAccessKeyId == accessKeyId)
    assert(awsCredentials.get.getAWSSecretKey == secretAccessKey)
  }

  it should "build Token Credentials" taggedAs SlowTest in {
    val accessKeyId = "myAccessKeyId"
    val secretAccessKey = "mySecretAccessKey"
    val sessionToken = "mySessionToken"
    val region = "myRegion"
    val credentialParams =
      CredentialParams(accessKeyId, secretAccessKey, sessionToken, "", region)

    val awsCredentials = new AWSTokenCredentials().buildCredentials(credentialParams)

    val awsSessionCredentials: BasicSessionCredentials =
      awsCredentials.get.asInstanceOf[BasicSessionCredentials]

    assert(awsSessionCredentials.getAWSAccessKeyId == accessKeyId)
    assert(awsSessionCredentials.getAWSSecretKey == secretAccessKey)
    assert(awsSessionCredentials.getSessionToken == sessionToken)
  }

  it should "build provided credentials" taggedAs SlowTest in {
    val secretAccessKey = "mySecretAccessKey"
    val sessionToken = "mySessionToken"
    val region = "myRegion"
    val credentialParams = CredentialParams("", secretAccessKey, sessionToken, "", region)

    val awsCredentials = new AWSTokenCredentials().buildCredentials(credentialParams)

    assert(awsCredentials.isDefined)
  }

  it should "build anonymous credentials" taggedAs SlowTest in {
    val accessKeyId = "anonymous"
    val region = "myRegion"
    val credentialParams = CredentialParams(accessKeyId, "", "", "", region)

    val awsCredentials = new AWSTokenCredentials().buildCredentials(credentialParams)

    val awsSessionCredentials: AnonymousAWSCredentials =
      awsCredentials.get.asInstanceOf[AnonymousAWSCredentials]

    assert(awsSessionCredentials.getAWSAccessKeyId == null)
    assert(awsSessionCredentials.getAWSSecretKey == null)
  }

  it should "build Profile Credentials" taggedAs SlowTest in {
    val profile = "myProfile"
    val region = "myRegion"

    val credentialParams = CredentialParams("", "", "", profile, region)

    val awsCredentials = new AWSTokenCredentials().buildCredentials(credentialParams)

    assert(awsCredentials.isDefined)
  }

}
