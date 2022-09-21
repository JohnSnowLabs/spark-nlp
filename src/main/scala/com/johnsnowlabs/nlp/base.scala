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

package com.johnsnowlabs.nlp

import org.apache.spark.ml.util.DefaultParamsReadable

package object base {

  type DocumentAssembler = com.johnsnowlabs.nlp.DocumentAssembler

  object DocumentAssembler extends DefaultParamsReadable[DocumentAssembler]

  type MultiDocumentAssembler = com.johnsnowlabs.nlp.MultiDocumentAssembler

  object MultiDocumentAssembler extends DefaultParamsReadable[MultiDocumentAssembler]

  type TokenAssembler = com.johnsnowlabs.nlp.TokenAssembler

  object TokenAssembler extends DefaultParamsReadable[TokenAssembler]

  type Doc2Chunk = com.johnsnowlabs.nlp.Doc2Chunk

  object Doc2Chunk extends DefaultParamsReadable[Doc2Chunk]

  type Chunk2Doc = com.johnsnowlabs.nlp.Chunk2Doc

  object Chunk2Doc extends DefaultParamsReadable[Chunk2Doc]

  type Finisher = com.johnsnowlabs.nlp.Finisher

  object Finisher extends DefaultParamsReadable[Finisher]

  type EmbeddingsFinisher = com.johnsnowlabs.nlp.EmbeddingsFinisher

  object EmbeddingsFinisher extends DefaultParamsReadable[EmbeddingsFinisher]

  type RecursivePipeline = com.johnsnowlabs.nlp.RecursivePipeline

  type LightPipeline = com.johnsnowlabs.nlp.LightPipeline

  type ImageAssembler = com.johnsnowlabs.nlp.ImageAssembler

  object ImageAssembler extends DefaultParamsReadable[ImageAssembler]

  type AudioAssembler = com.johnsnowlabs.nlp.AudioAssembler

  object AudioAssembler extends DefaultParamsReadable[AudioAssembler]

}
