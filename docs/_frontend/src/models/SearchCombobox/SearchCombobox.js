import classNames from 'classnames';
import { useCombobox, useMultipleSelection } from 'downshift';
import React, { useState, useEffect } from 'react';
import debounce from 'lodash.debounce';
import styles from './SearchCombobox.module.css';

const SearchCombobox = ({
  label,
  autoComplete,
  initialSelectedItems,
  onChange,
}) => {
  const [inputValue, setInputValue] = useState('');
  const [items, setItems] = useState([]);
  const [loading, setLoadingMessage] = useState('Start Typing ...');

  const {
    getSelectedItemProps,
    getDropdownProps,
    addSelectedItem,
    removeSelectedItem,
    selectedItems,
  } = useMultipleSelection({
    initialSelectedItems,
    onSelectedItemsChange: ({ selectedItems }) => {
      onChange(selectedItems);
    },
  });

  useEffect(() => {
    if (inputValue) {
      const fetchItems = (input, selectedItem) => {
        autoComplete(input)
          .then((f) => {
            if (f.ok) {
              return f.json();
            }
          })
          .then((result) => {
            setItems(
              result.data.filter(
                (item) => !selectedItems.includes(item) && item !== selectedItem
              )
            );
          })
          .finally(() => setLoadingMessage(''));
      };

      setLoadingMessage('Loading ...');
      const findItemsButChill = debounce(fetchItems, 350);
      findItemsButChill(inputValue);
    } else {
      setItems([]);
      setLoadingMessage('Start typing ...');
    }
  }, [inputValue, selectedItems]);

  const {
    isOpen,
    getToggleButtonProps,
    getLabelProps,
    getMenuProps,
    getInputProps,
    getComboboxProps,
    highlightedIndex,
    getItemProps,
  } = useCombobox({
    inputValue,
    defaultHighlightedIndex: 0, // after selection, highlight the first item.
    selectedItem: null,
    items: items,
    onInputValueChange: () => {
      setInputValue(inputValue);
    },
    stateReducer: (state, actionAndChanges) => {
      const { changes, type } = actionAndChanges;
      switch (type) {
        case useCombobox.stateChangeTypes.InputKeyDownEnter:
        case useCombobox.stateChangeTypes.ItemClick:
          return {
            ...changes,
            isOpen: false, // keep the menu open after selection.
          };
      }
      return changes;
    },
    onStateChange: ({ inputValue, type, selectedItem, ...other }) => {
      switch (type) {
        case useCombobox.stateChangeTypes.InputChange:
          setInputValue(inputValue);
          break;
        case useCombobox.stateChangeTypes.InputKeyDownEnter:
        case useCombobox.stateChangeTypes.ItemClick:
        case useCombobox.stateChangeTypes.InputBlur:
          if (selectedItem) {
            //reset items
            setInputValue('');
            addSelectedItem(selectedItem);
          }
          break;
        default:
          break;
      }
    },
  });

  return (
    <div>
      <label {...getLabelProps()} className={styles.label}>
        {label}
      </label>
      <div className={styles.comboboxWrapper}>
        {selectedItems.map((selectedItem, index) => (
          <span
            className={styles.selectedItem}
            key={`selected-item-${index}`}
            {...getSelectedItemProps({ selectedItem, index })}
          >
            <span className={styles.selectedItem__value}>{selectedItem}</span>

            <span
              className={styles.selectedItem__x}
              onClick={(e) => {
                e.stopPropagation();
                removeSelectedItem(selectedItem);
              }}
            >
              &#10005;
            </span>
          </span>
        ))}
        <div className={styles.combobox} {...getComboboxProps()}>
          <input
            {...getInputProps(getDropdownProps({ preventKeyAction: isOpen }))}
            className={styles.combobox__input}
          />
          <button
            {...getToggleButtonProps()}
            aria-label={'toggle menu'}
            className={styles.combobox__toggle}
          />
        </div>
      </div>
      <div className={styles.menuWrapper}>
        {isOpen &&
          (items.length > 0 ? (
            <ul {...getMenuProps()} className={styles.menu}>
              {items.map((item, index) => (
                <li
                  className={styles.menu__item}
                  style={
                    highlightedIndex === index
                      ? { backgroundColor: '#bde4ff' }
                      : {}
                  }
                  key={`${item}${index}`}
                  {...getItemProps({ item, index })}
                >
                  {item}
                </li>
              ))}
            </ul>
          ) : (
            <ul {...getMenuProps()} className={styles.menu}>
              {loading ? (
                <li className={classNames(styles.menu__item, styles.isEmpty)}>
                  {loading}
                </li>
              ) : (
                <li className={classNames(styles.menu__item, styles.isEmpty)}>
                  No options
                </li>
              )}
            </ul>
          ))}
      </div>
    </div>
  );
};

export default SearchCombobox;
