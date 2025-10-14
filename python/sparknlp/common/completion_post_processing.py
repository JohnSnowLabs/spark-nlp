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
from pyspark.ml.param import Param, Params, TypeConverters


class CompletionPostProcessing:
    removeThinkingTag = Param(
        Params._dummy(),
        "removeThinkingTag",
        "Set a thinking tag (e.g. think) to be removed from output. Will match <TAG>...</TAG>",
        typeConverter=TypeConverters.toString,
    )

    def setRemoveThinkingTag(self, value: str):
        """Set a thinking tag (e.g. `think`) to be removed from output.
        Will produce the regex: `(?s)<$TAG>.+?</$TAG>`
        """
        self._set(removeThinkingTag=value)
        return self

    def getRemoveThinkingTag(self):
        """Get the thinking tag to be removed from output."""
        value = None
        if self.removeThinkingTag in self._paramMap:
            value = self._paramMap[self.removeThinkingTag]
        return value
