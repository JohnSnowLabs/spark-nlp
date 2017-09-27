package com.jsl.ml.crf


case class Attr
(
   val id: Int,
   val name: String,
   val isNumerical: Boolean = false
)

case class Transition
(
   val stateFrom: Int,
   val stateTo: Int
)

case class AttrFeature
(
  val id: Int,
  val attrId: Int,
  val label: Int
)
