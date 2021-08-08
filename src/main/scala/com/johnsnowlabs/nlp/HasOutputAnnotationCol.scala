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

import org.apache.spark.ml.param.{Param, Params}

trait HasOutputAnnotationCol extends Params {

  protected final val outputCol: Param[String] =
    new Param(this, "outputCol", "the output annotation column")

  /** Overrides annotation column name when transforming */
  final def setOutputCol(value: String): this.type = set(outputCol, value)

  /** Gets annotation column name going to generate */
  final def getOutputCol: String = $(outputCol)

}
