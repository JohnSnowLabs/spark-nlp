/*
 * Copyright 2017-2025 John Snow Labs
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
package com.johnsnowlabs.partition

import com.johnsnowlabs.nlp.ParamsAndFeaturesWritable
import org.apache.spark.ml.param.Param

trait HasSemanticChunkerProperties extends ParamsAndFeaturesWritable {

  val chunkingStrategy = new Param[String](this, "chunkingStrategy", "Set the chunking strategy")

  def setChunkingStrategy(value: String): this.type = set(chunkingStrategy, value)

  val maxCharacters =
    new Param[Int](this, "maxCharacters", "Set the maximum number of characters")

  def setMaxCharacters(value: Int): this.type = set(maxCharacters, value)

  val newAfterNChars =
    new Param[Int](this, "newAfterNChars", "Insert a new chunk after N characters")

  def setNewAfterNChars(value: Int): this.type = set(newAfterNChars, value)

  val overlap =
    new Param[Int](this, "overlap", "Set the number of overlapping characters between chunks")

  def setOverlap(value: Int): this.type = set(overlap, value)

  setDefault(chunkingStrategy -> "", maxCharacters -> 100, newAfterNChars -> -1, overlap -> 0)

}
