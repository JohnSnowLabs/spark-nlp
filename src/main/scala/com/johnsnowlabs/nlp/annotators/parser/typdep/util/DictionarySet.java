/*
 * Copyright 2017-2021 John Snow Labs
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

package com.johnsnowlabs.nlp.annotators.parser.typdep.util;

import gnu.trove.map.TIntIntMap;
import gnu.trove.map.hash.TIntIntHashMap;

import java.io.Serializable;

public class DictionarySet implements Serializable {

    private static final long serialVersionUID = 1L;

    public enum DictionaryTypes
    {
        POS,
        WORD,
        DEP_LABEL,
        WORD_VEC,
        TYPE_END
    }

    private Dictionary[] dictionaries;

    private boolean isCounting;

    public Dictionary[] getDictionaries() {
        return dictionaries;
    }

    public boolean isCounting() {
        return isCounting;
    }

    private TIntIntMap[] counters;

    public DictionarySet()
    {
        isCounting = false;
        int indexDictionaryTypes = DictionaryTypes.TYPE_END.ordinal();
        dictionaries = new Dictionary[indexDictionaryTypes];
        for (int i = 0; i < dictionaries.length; ++i) {
            dictionaries[i] = new Dictionary();
        }
    }

    public int lookupIndex(DictionaryTypes tag, String item)
    {
        int id = dictionaries[tag.ordinal()].lookupIndex(item);

        if (isCounting && id > 0) {
            counters[tag.ordinal()].putIfAbsent(id, 0);
            counters[tag.ordinal()].increment(id);
        }

        return id <= 0 ? 1 : id;
    }

    public int getDictionarySize(DictionaryTypes tag)
    {
        int indexTag = tag.ordinal();
        return dictionaries[indexTag].dictionarySize();
    }

    public void stopGrowth(DictionaryTypes tag)
    {
        dictionaries[tag.ordinal()].stopGrowth();
    }

    public Dictionary getDictionary(DictionaryTypes tag)
    {
        return dictionaries[tag.ordinal()];
    }

    public void setCounters()
    {
        isCounting = true;
        counters = new TIntIntHashMap[dictionaries.length];
        for (int i = 0; i < dictionaries.length; ++i)
            counters[i] = new TIntIntHashMap();
    }

    public void closeCounters()
    {
        isCounting = false;
        counters = null;
    }

}
