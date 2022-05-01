import React, { useMemo } from 'react';
import styles from './RadioGroup.module.css';
import RadioGroupContext from './RadioGroupContext';

const RadioGroup = ({ name, value, onChange, isDisabled, children }) => {
  const context = useMemo(() => {
    return {
      name,
      value,
      onChange,
      isDisabled,
    };
  }, [name, value, onChange, isDisabled]);
  return (
    <div className={styles.root}>
      <RadioGroupContext.Provider value={context}>
        {children}
      </RadioGroupContext.Provider>
    </div>
  );
};

export default RadioGroup;
