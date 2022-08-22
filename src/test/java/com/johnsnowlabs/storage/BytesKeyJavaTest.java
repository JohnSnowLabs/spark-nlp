package com.johnsnowlabs.storage;

import org.junit.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.*;

public class BytesKeyJavaTest {

    @Test
    public void shouldReturnValueGivenAKey() {
        BytesKey key1 = new BytesKey(new byte[]{1, 2, 3});
        BytesKey key2 = new BytesKey(new byte[]{50, 48, 51});
        Map<BytesKey, String> map = new HashMap<>();
        map.put(key1, "value1");
        map.put(key2, "value2");

        String retrievedValue1 = map.get(key1);
        String retrievedValue2 = map.get(key2);
        String retrievedValue3 = map.get(new BytesKey(new byte[]{1, 2, 3}));

        assertEquals("value1", retrievedValue1);
        assertEquals("value2", retrievedValue2);
        assertEquals("value1", retrievedValue3);
    }

}
