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
package com.johnsnowlabs.client

import com.johnsnowlabs.client.aws.AWSClient
import com.johnsnowlabs.client.gcp.GCPClient

class CloudManager(parameters: Map[String, String] = Map.empty) {

  def getClientInstance(uri: String): CloudClient = {
    uri match {
      case s3Uri if s3Uri.startsWith("s3://") || s3Uri.startsWith("s3a://") =>
        new AWSClient(parameters)
      case gcpUri if gcpUri.startsWith("gs://") => {
        new GCPClient(parameters)
      }
      //      case azureUri
      //          if azureUri.startsWith("https://") && azureUri.contains(".blob.core.windows.net/") => "Azure"
      case _ =>
        throw new IllegalArgumentException(s"Unsupported URI scheme: $uri")
    }
  }

}
