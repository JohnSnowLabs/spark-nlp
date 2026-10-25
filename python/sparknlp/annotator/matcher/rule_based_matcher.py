#  Copyright 2017-2026 John Snow Labs
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Contains classes for the RuleBasedMatcher."""


import json

from sparknlp.common import *


class _RuleBasedMatcherParams:
    attributeColumns = Param(
        Params._dummy(),
        "attributeColumns",
        "Attribute to input column mappings encoded as ATTRIBUTE=column",
        typeConverter=TypeConverters.toListString,
    )

    alignmentMode = Param(
        Params._dummy(),
        "alignmentMode",
        "Annotation alignment mode: STRICT or POSITIONAL",
        typeConverter=TypeConverters.toString,
    )

    overlapStrategy = Param(
        Params._dummy(),
        "overlapStrategy",
        "Overlap strategy: ALL, FIRST, LONGEST, or PRIORITY_LONGEST",
        typeConverter=TypeConverters.toString,
    )

    def setInputCols(self, *value):
        """Sets input annotation columns.

        The RuleBasedMatcher accepts a variable number of annotation columns.
        """
        if type(value[0]) == str or type(value[0]) == list:
            if len(value) == 1 and type(value[0]) == list:
                return self._set(inputCols=value[0])
            else:
                return self._set(inputCols=list(value))
        else:
            raise TypeError("InputCols datatype not supported. It must be either str or list")

    def setAttributeColumns(self, value):
        """Sets attribute-to-column mappings.

        Parameters
        ----------
        value : dict or list
            Either ``{"POS": "pos"}`` or ``["POS=pos"]`` style mappings.
        """
        if isinstance(value, dict):
            for attr, col in value.items():
                if not isinstance(attr, str) or not attr.strip():
                    raise ValueError("attributeColumns dict keys must be non-empty strings")
                if not isinstance(col, str) or not col.strip():
                    raise ValueError("attributeColumns dict values must be non-empty column names")
            return self._set(attributeColumns=[f"{attr}={col}" for attr, col in value.items()])
        if isinstance(value, list):
            for entry in value:
                if not isinstance(entry, str) or "=" not in entry:
                    raise ValueError(
                        "attributeColumns list entries must use ATTRIBUTE=column string format"
                    )
                attr, col = entry.split("=", 1)
                if not attr.strip() or not col.strip():
                    raise ValueError(
                        "attributeColumns list entries must use non-empty ATTRIBUTE=column values"
                    )
            return self._set(attributeColumns=value)
        raise TypeError("attributeColumns must be a dict or list")

    def setAlignmentMode(self, value):
        """Sets annotation alignment mode, either ``STRICT`` or ``POSITIONAL``."""
        normalized = value.upper()
        if normalized not in ("STRICT", "POSITIONAL"):
            raise ValueError("alignmentMode must be either STRICT or POSITIONAL")
        return self._set(alignmentMode=normalized)

    def setOverlapStrategy(self, value):
        """Sets overlap strategy: ``ALL``, ``FIRST``, ``LONGEST``, or ``PRIORITY_LONGEST``."""
        normalized = value.upper()
        if normalized not in ("ALL", "FIRST", "LONGEST", "PRIORITY_LONGEST"):
            raise ValueError("overlapStrategy must be one of ALL, FIRST, LONGEST, PRIORITY_LONGEST")
        return self._set(overlapStrategy=normalized)


class RuleBasedMatcher(_RuleBasedMatcherParams, AnnotatorApproach):
    """Rule-based token matcher over multiple Spark NLP annotation columns.

    Rules are supplied as JSON or JSONL. Each token pattern can combine
    attributes from token text, POS, lemmas, NER, dependency metadata, or custom
    mapped annotation columns. Matches are emitted as ``CHUNK`` annotations.

    ================================ ======================
    Input Annotation types           Output Annotation type
    ================================ ======================
    ``DOCUMENT, TOKEN, ...``         ``CHUNK``
    ================================ ======================
    """

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT, AnnotatorType.TOKEN]

    outputAnnotatorType = AnnotatorType.CHUNK

    rules = Param(
        Params._dummy(),
        "rules",
        "Inline JSON or JSONL rule definitions",
        typeConverter=TypeConverters.toString,
    )

    rulesResource = Param(
        Params._dummy(),
        "rulesResource",
        "External JSON or JSONL rule resource",
        typeConverter=TypeConverters.identity,
    )

    @keyword_only
    def __init__(self):
        super(RuleBasedMatcher, self).__init__(
            classname="com.johnsnowlabs.nlp.annotators.matcher.RuleBasedMatcher"
        )
        self._setDefault(attributeColumns=[], alignmentMode="STRICT", overlapStrategy="ALL")

    def _create_model(self, java_model):
        return RuleBasedMatcherModel(java_model=java_model)

    def setRules(self, value):
        """Sets inline JSON/JSONL rule definitions.

        Parameters
        ----------
        value : str, dict, or list
            JSON/JSONL string, one rule dict, or a list of rule dicts.
        """
        if self.isSet(self.rulesResource):
            raise ValueError("Only one of rules or rulesResource can be set")
        if isinstance(value, str):
            rules = value
        elif isinstance(value, (dict, list)):
            rules = json.dumps(value)
        else:
            raise TypeError("rules must be a JSON string, dict, or list")
        return self._set(rules=rules)

    def setRulesResource(self, path, read_as=ReadAs.TEXT, options={"format": "text"}):
        """Sets an external JSON or JSONL rules resource."""
        if self.isSet(self.rules):
            raise ValueError("Only one of rules or rulesResource can be set")
        return self._set(rulesResource=ExternalResource(path, read_as, options.copy()))


class RuleBasedMatcherModel(_RuleBasedMatcherParams, AnnotatorModel):
    """Instantiated model of :class:`.RuleBasedMatcher`."""

    name = "RuleBasedMatcherModel"

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT, AnnotatorType.TOKEN]

    outputAnnotatorType = AnnotatorType.CHUNK

    inputColumnTypes = Param(
        Params._dummy(),
        "inputColumnTypes",
        "Input column annotator types encoded as column=annotatorType",
        typeConverter=TypeConverters.toListString,
    )

    rulesJson = Param(
        Params._dummy(),
        "rulesJson",
        "Normalized JSON rule definitions",
        typeConverter=TypeConverters.toString,
    )

    def __init__(
            self,
            classname="com.johnsnowlabs.nlp.annotators.matcher.RuleBasedMatcherModel",
            java_model=None):
        super(RuleBasedMatcherModel, self).__init__(
            classname=classname,
            java_model=java_model,
        )
