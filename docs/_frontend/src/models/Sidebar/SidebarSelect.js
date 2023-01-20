import React from 'react';
import { useSelect } from 'downshift';
import classNames from 'classnames';
import styles from './SidebarSelect.module.css';

const SidebarSelect = ({ label, items, selectedItem, onChange }) => {
  const {
    isOpen,
    getToggleButtonProps,
    getLabelProps,
    getMenuProps,
    highlightedIndex,
    getItemProps,
  } = useSelect({
    items,
    itemToString: (item) => item,
    selectedItem,
    onSelectedItemChange: ({ selectedItem: nextSelectedItem }) =>
      onChange(nextSelectedItem),
  });

  return (
    <div>
      <label {...getLabelProps()} className={styles.label}>
        {label}
      </label>
      <div className={styles.select}>
        {selectedItem ? (
          <>
            <span className={styles.select__value}>{selectedItem}</span>
            <button
              className={classNames(
                styles.select__clear,
                'fa',
                'fa-times-circle'
              )}
              onClick={() => {
                onChange(null);
              }}
            />
          </>
        ) : (
          <button
            {...getToggleButtonProps()}
            aria-label={'toggle menu'}
            className={styles.select__toggle}
          ></button>
        )}
      </div>
      {selectedItem ? null : (
        <div className={styles.menuWrapper}>
          {isOpen && (
            <ul {...getMenuProps()} className={styles.menu}>
              {items.map((item, index) => (
                <li
                  className={classNames(styles.menu__item)}
                  style={
                    highlightedIndex === index
                      ? { backgroundColor: '#bde4ff' }
                      : {}
                  }
                  key={`${item.value}${index}`}
                  {...getItemProps({ item, index })}
                >
                  {item}
                </li>
              ))}
              {items.length === 0 && (
                <li className={classNames(styles.menu__item, styles.isEmpty)}>
                  No options
                </li>
              )}
            </ul>
          )}
        </div>
      )}
    </div>
  );
};

export default SidebarSelect;
