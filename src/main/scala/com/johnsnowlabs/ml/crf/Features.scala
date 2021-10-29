/*
 * Copyright 2017-2019 John Snow Labs
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

package com.johnsnowlabs.ml.crf


case class Attr
(
   id: Int,
   name: String,
   isNumerical: Boolean = false
)

case class Transition
(
   stateFrom: Int,
   stateTo: Int
)

case class AttrFeature
(
  id: Int,
  attrId: Int,
  label: Int
)
