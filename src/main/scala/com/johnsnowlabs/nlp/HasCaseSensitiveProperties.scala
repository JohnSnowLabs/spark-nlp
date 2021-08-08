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

import org.apache.spark.ml.param.BooleanParam

trait HasCaseSensitiveProperties extends ParamsAndFeaturesWritable {

  /** Whether to ignore case in index lookups (Default depends on model)
   *
   * @group param
   */
  val caseSensitive = new BooleanParam(this, "caseSensitive", "Whether to ignore case in index lookups")

  setDefault(caseSensitive, false)

  /** @group getParam */
  def getCaseSensitive: Boolean = $(caseSensitive)

  /** @group setParam */
  def setCaseSensitive(value: Boolean): this.type = set(this.caseSensitive, value)

}
