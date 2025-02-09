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
"""Contains classes for Extractor."""
from sparknlp.common import *

class Extractor(AnnotatorModel):
    name = "Extractor"

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT]

    outputAnnotatorType = AnnotatorType.CHUNK

    emailDateTimeTzPattern = Param(Params._dummy(),
                                   "emailDateTimeTzPattern",
                                   "Specifies the date-time pattern for email timestamps, including time zone formatting.",
                                   typeConverter=TypeConverters.toString)

    emailAddress = Param(
        Params._dummy(),
        "emailAddress",
        "Specifies the pattern for email addresses.",
        typeConverter=TypeConverters.toString
    )

    ipAddressPattern = Param(
        Params._dummy(),
        "ipAddressPattern",
        "Specifies the pattern for IP addresses.",
        typeConverter=TypeConverters.toString
    )

    ipAddressNamePattern = Param(
        Params._dummy(),
        "ipAddressNamePattern",
        "Specifies the pattern for IP addresses with names.",
        typeConverter=TypeConverters.toString
    )

    mapiIdPattern = Param(
        Params._dummy(),
        "mapiIdPattern",
        "Specifies the pattern for MAPI IDs.",
        typeConverter=TypeConverters.toString
    )

    usPhoneNumbersPattern = Param(
        Params._dummy(),
        "usPhoneNumbersPattern",
        "Specifies the pattern for US phone numbers.",
        typeConverter=TypeConverters.toString
    )

    imageUrlPattern = Param(
        Params._dummy(),
        "imageUrlPattern",
        "Specifies the pattern for image URLs.",
        typeConverter=TypeConverters.toString
    )

    textPattern = Param(
        Params._dummy(),
        "textPattern",
        "Specifies the pattern for text after and before.",
        typeConverter=TypeConverters.toString
    )

    extractorMode = Param(
        Params._dummy(),
        "extractorMode",
        "possible values: " +
        "email_date, email_address, ip_address, ip_address_name, mapi_id, us_phone_numbers, image_urls, bullets, text_after, text_before",
        typeConverter=TypeConverters.toString
    )

    index = Param(
        Params._dummy(),
        "index",
        "Specifies the index of the pattern to extract in text after or before",
        typeConverter=TypeConverters.toInt
    )

    def setEmailDateTimeTzPattern(self, value):
        """Sets specifies the date-time pattern for email timestamps, including time zone formatting.

        Parameters
        ----------
        value : str
            Specifies the date-time pattern for email timestamps, including time zone formatting.
        """
        return self._set(emailDateTimeTzPattern=value)

    def setEmailAddress(self, value):
        """Sets the pattern for email addresses.

        Parameters
        ----------
        value : str
            Specifies the pattern for email addresses.
        """
        return self._set(emailAddress=value)

    def setIpAddressPattern(self, value):
        """Sets the pattern for IP addresses.

        Parameters
        ----------
        value : str
            Specifies the pattern for IP addresses.
        """
        return self._set(ipAddressPattern=value)

    def setIpAddressNamePattern(self, value):
        """Sets the pattern for IP addresses with names.

        Parameters
        ----------
        value : str
            Specifies the pattern for IP addresses with names.
        """
        return self._set(ipAddressNamePattern=value)

    def setMapiIdPattern(self, value):
        """Sets the pattern for MAPI IDs.

        Parameters
        ----------
        value : str
            Specifies the pattern for MAPI IDs.
        """
        return self._set(mapiIdPattern=value)

    def setUsPhoneNumbersPattern(self, value):
        """Sets the pattern for US phone numbers.

        Parameters
        ----------
        value : str
            Specifies the pattern for US phone numbers.
        """
        return self._set(usPhoneNumbersPattern=value)

    def setImageUrlPattern(self, value):
        """Sets the pattern for image URLs.

        Parameters
        ----------
        value : str
            Specifies the pattern for image URLs.
        """
        return self._set(imageUrlPattern=value)

    def setTextPattern(self, value):
        """Sets the pattern for text after and before.

        Parameters
        ----------
        value : str
            Specifies the pattern for text after and before.
        """
        return self._set(textPattern=value)

    def setExtractorMode(self, value):
        return self._set(extractorMode=value)

    def setIndex(self, value):
        """Sets the index of the pattern to extract in text after or before.

        Parameters
        ----------
        value : int
            Specifies the index of the pattern to extract in text after or before.
        """
        return self._set(index=value)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.cleaners.Extractor", java_model=None):
        super(Extractor, self).__init__(
            classname=classname,
            java_model=java_model
        )