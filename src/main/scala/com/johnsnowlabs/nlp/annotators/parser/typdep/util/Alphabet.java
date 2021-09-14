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

import gnu.trove.map.hash.TLongIntHashMap;

import java.io.Serializable;

public class Alphabet implements Serializable
{
    // Serialization
    private static final long serialVersionUID = 1;
    private TLongIntHashMap map;
    private int numEntries;
    private boolean growthStopped = false;

    private Alphabet (int capacity)
    {
        this.map = new TLongIntHashMap(capacity);
        numEntries = 0;
    }

    public Alphabet ()
    {
        this (10000);
    }

    /** Return -1 if entry isn't present. */
    public int lookupIndex (long entry, int value)
    {
        int ret = map.get(entry);
        if (ret <= 0 && !growthStopped) {
            numEntries++;
            ret = value + 1;
            map.put (entry, ret);
        }
        return ret - 1;	// feature id should be 0-based
    }

    /** Return -1 if entry isn't present. */
    public int lookupIndex (long entry)
    {
        int ret = map.get(entry);
        if (ret <= 0 && !growthStopped) {
            numEntries++;
            ret = numEntries;
            map.put (entry, ret);
        }
        return ret - 1;	// feature id should be 0-based
    }

    public void stopGrowth ()
    {
        growthStopped = true;
    }
}
