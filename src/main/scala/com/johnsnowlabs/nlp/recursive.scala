/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.nlp

import org.apache.spark.ml.{Pipeline, PipelineModel}

package object recursive {

  implicit def p2recursive(pipeline: Pipeline): RecursivePipeline =
    new RecursivePipeline(pipeline)

  implicit def pm2recursive(pipelineModel: PipelineModel): RecursivePipelineModel =
    new RecursivePipelineModel(pipelineModel.uid, pipelineModel)

  implicit def pm2light(pipelineModel: PipelineModel): LightPipeline =
    new LightPipeline(pipelineModel)

  implicit class Recursive(p: Pipeline) {
    def recursive: RecursivePipeline = {
      new RecursivePipeline(p)
    }
  }

  implicit class RecursiveModel(p: PipelineModel) {
    def recursive: RecursivePipelineModel = {
      new RecursivePipelineModel(p.uid, p)
    }
  }

}
