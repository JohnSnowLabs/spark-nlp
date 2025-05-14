#  Copyright 2017-2025 John Snow Labs
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
"""Contains classes for Cleaner."""
from sparknlp.annotator import MarianTransformer
from sparknlp.common import *

class Cleaner(MarianTransformer):
    name = "Cleaner"

    inputAnnotatorTypes = [AnnotatorType.TOKEN]

    outputAnnotatorType = AnnotatorType.CHUNK

    encoding = Param(Params._dummy(),
                   "encoding",
                   "The encoding to be used for decoding the byte string (default is utf-8)",
                   typeConverter=TypeConverters.toString)

    cleanPrefixPattern = Param(Params._dummy(),
                     "cleanPrefixPattern",
                     "The pattern for the prefix. Can be a simple string or a regex pattern.",
                     typeConverter=TypeConverters.toString)

    cleanPostfixPattern = Param(Params._dummy(),
                               "cleanPostfixPattern",
                               "The pattern for the postfix. Can be a simple string or a regex pattern.",
                               typeConverter=TypeConverters.toString)

    cleanerMode = Param(
        Params._dummy(),
        "cleanerMode",
        "possible values: " +
        "clean, bytes_string_to_string, clean_non_ascii_chars, clean_ordered_bullets, clean_postfix, clean_prefix, remove_punctuation, replace_unicode_quotes",
        typeConverter=TypeConverters.toString
    )

    extraWhitespace = Param(Params._dummy(),
                    "extraWhitespace",
                    "Whether to remove extra whitespace.",
                    typeConverter=TypeConverters.toBoolean)

    dashes = Param(Params._dummy(),
                "dashes",
                "Whether to handle dashes in text.",
                typeConverter=TypeConverters.toBoolean)

    bullets = Param(Params._dummy(),
                   "bullets",
                   "Whether to handle bullets in text.",
                   typeConverter=TypeConverters.toBoolean)

    trailingPunctuation = Param(Params._dummy(),
                    "trailingPunctuation",
                    "Whether to remove trailing punctuation from text.",
                    typeConverter=TypeConverters.toBoolean)

    lowercase = Param(Params._dummy(),
                "lowercase",
                "Whether to convert text to lowercase.",
                typeConverter=TypeConverters.toBoolean)

    ignoreCase = Param(Params._dummy(),
                      "ignoreCase",
                      "If true, ignores case in the pattern.",
                      typeConverter=TypeConverters.toBoolean)

    strip = Param(Params._dummy(),
               "strip",
               "If true, removes leading or trailing whitespace from the cleaned string.",
               typeConverter=TypeConverters.toBoolean)

    def setEncoding(self, value):
        """Sets the encoding to be used for decoding the byte string (default is utf-8).

        Parameters
        ----------
        value : str
            The encoding to be used for decoding the byte string (default is utf-8)
        """
        return self._set(encoding=value)

    def setCleanPrefixPattern(self, value):
        """Sets the pattern for the prefix. Can be a simple string or a regex pattern.

        Parameters
        ----------
        value : str
            The pattern for the prefix. Can be a simple string or a regex pattern.
        """
        return self._set(cleanPrefixPattern=value)

    def setCleanPostfixPattern(self, value):
        """Sets the pattern for the postfix. Can be a simple string or a regex pattern.

        Parameters
        ----------
        value : str
            The pattern for the postfix. Can be a simple string or a regex pattern.
        """
        return self._set(cleanPostfixPattern=value)

    def setCleanerMode(self, value):
        """Sets the cleaner mode.

        Possible values:
            clean, bytes_string_to_string, clean_non_ascii_chars, clean_ordered_bullets,
            clean_postfix, clean_prefix, remove_punctuation, replace_unicode_quotes

        Parameters
        ----------
        value : str
            The mode for cleaning operations.
        """
        return self._set(cleanerMode=value)

    def setExtraWhitespace(self, value):
        """Sets whether to remove extra whitespace.

        Parameters
        ----------
        value : bool
            Whether to remove extra whitespace.
        """
        return self._set(extraWhitespace=value)

    def setDashes(self, value):
        """Sets whether to handle dashes in text.

        Parameters
        ----------
        value : bool
            Whether to handle dashes in text.
        """
        return self._set(dashes=value)

    def setBullets(self, value):
        """Sets whether to handle bullets in text.

        Parameters
        ----------
        value : bool
            Whether to handle bullets in text.
        """
        return self._set(bullets=value)

    def setTrailingPunctuation(self, value):
        """Sets whether to remove trailing punctuation from text.

        Parameters
        ----------
        value : bool
            Whether to remove trailing punctuation from text.
        """
        return self._set(trailingPunctuation=value)

    def setLowercase(self, value):
        """Sets whether to convert text to lowercase.

        Parameters
        ----------
        value : bool
            Whether to convert text to lowercase.
        """
        return self._set(lowercase=value)

    def setIgnoreCase(self, value):
        """Sets whether to ignore case in the pattern.

        Parameters
        ----------
        value : bool
            If true, ignores case in the pattern.
        """
        return self._set(ignoreCase=value)

    def setStrip(self, value):
        """Sets whether to remove leading or trailing whitespace from the cleaned string.

        Parameters
        ----------
        value : bool
            If true, removes leading or trailing whitespace from the cleaned string.
        """
        return self._set(strip=value)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.cleaners.Cleaner", java_model=None):
        super(Cleaner, self).__init__(
            classname=classname,
            java_model=java_model
        )