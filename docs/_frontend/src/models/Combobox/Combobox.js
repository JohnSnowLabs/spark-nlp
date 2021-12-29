import classNames from 'classnames';
import { useCombobox, useMultipleSelection } from 'downshift';
import React, { useState } from 'react';
import styles from './Combobox.module.css';

const Combobox = ({ label, items, initialSelectedItems, onChange }) => {
  const [inputValue, setInputValue] = useState('');
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
  const getFilteredItems = () =>
    items.filter(
      (item) =>
        selectedItems.indexOf(item) < 0 &&
        item.toLowerCase().startsWith(inputValue.toLowerCase())
    );
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
    items: getFilteredItems(),
    stateReducer: (state, actionAndChanges) => {
      const { changes, type } = actionAndChanges;
      switch (type) {
        case useCombobox.stateChangeTypes.InputKeyDownEnter:
        case useCombobox.stateChangeTypes.ItemClick:
          return {
            ...changes,
            isOpen: true, // keep the menu open after selection.
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
            setInputValue('');
            addSelectedItem(selectedItem);
          }
          break;
        default:
          break;
      }
    },
  });

  const filteredItems = getFilteredItems(items);

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
          (filteredItems.length > 0 ? (
            <ul {...getMenuProps()} className={styles.menu}>
              {filteredItems.map((item, index) => (
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
              <li className={classNames(styles.menu__item, styles.isEmpty)}>
                No options
              </li>
            </ul>
          ))}
      </div>
    </div>
  );
};

export default Combobox;
