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
