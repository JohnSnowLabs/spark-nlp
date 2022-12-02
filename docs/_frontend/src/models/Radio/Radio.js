import classNames from 'classnames';
import React, { useContext } from 'react';
import { RadioGroupContext } from '../RadioGroup';
import styles from './Radio.module.css';

const Radio = ({ value, className, isDisabled, children }) => {
  const radioGroupContext = useContext(RadioGroupContext);

  return (
    <label
      className={classNames(styles.root, className, {
        [styles.isDisabled]: isDisabled || radioGroupContext.isDisabled,
      })}
    >
      <input
        type="radio"
        name={radioGroupContext.name}
        value={value}
        checked={radioGroupContext.value === value}
        className={styles.input}
        onChange={(e) => radioGroupContext.onChange(e.target.value)}
        disabled={isDisabled || radioGroupContext.isDisabled}
      />
      {children}
    </label>
  );
};

export default Radio;
