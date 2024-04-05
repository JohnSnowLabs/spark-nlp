/*
 * Copyright 2017-2022 John Snow Labs
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

import java.io.Serializable;
import java.util.HashMap;

public class Dictionary implements Serializable {

    // Serialization
    private static final long serialVersionUID = 1;

    private HashMap<String, Integer> map;

    public HashMap<String, Integer> getMap() {
        return map;
    }

    public void setMap(HashMap<String, Integer> map) {
        this.map = map;
    }

    private int numEntries;
    private boolean growthStopped = false;
    private String mapAsString;
    public int getNumEntries() {
        return numEntries;
    }

    public boolean isGrowthStopped() {
        return growthStopped;
    }

    public void setNumEntries(int numEntries) {
        this.numEntries = numEntries;
    }

    public void setGrowthStopped(boolean growthStopped) {
        this.growthStopped = growthStopped;
    }

    public void setMapAsString(String mapAsString) {
        this.mapAsString = mapAsString;
    }
    public String getMapAsString() {
        return mapAsString;
    }

    private Dictionary(int capacity) {
        this.map = new HashMap<>(capacity);
        numEntries = 0;
    }

    Dictionary() {
        this(10000);
    }

    /**
     * Return -1 (in old trove version) or 0 (in trove current verion) if entry isn't present.
     */
    public int lookupIndex(String entry) {
        if (entry == null) {
            return 0;
        }
        int ret = (map.get(entry) == null) ? 0 : map.get(entry);
        if (ret <= 0 && !growthStopped) {
            numEntries++;
            ret = numEntries;
            map.put(entry, ret);
        }
        return ret;
    }

    public String[] newToArray () {
        return map.keySet().toArray(new String[0]);
    }

    public int dictionarySize() {
        return numEntries;
    }

    void stopGrowth() {
        growthStopped = true;
    }

}
