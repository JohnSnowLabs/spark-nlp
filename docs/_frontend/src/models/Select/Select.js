import { createElement as e, Children } from 'react';
import './Select.css';

const Select = ({
  name,
  value,
  renderValue = (v) => v,
  className,
  onChange,
  disabled,
  children,
}) => {
  const handleChange = (e) => {
    if (typeof onChange === 'function') {
      onChange(e);
    }
  };

  const handleClear = () => {
    if (typeof onChange === 'function') {
      onChange({
        target: { value: '' },
      });
    }
  };

  if (value) {
    return e(
      'div',
      {
        className: [
          'select',
          'select--value',
          disabled && 'select--disabled',
          className,
        ]
          .filter(Boolean)
          .join(' '),
        title: renderValue(value),
      },
      [
        e(
          'span',
          { key: 'value', className: 'select__value' },
          renderValue(value)
        ),
        e('button', {
          key: 'clear',
          className: 'select__clear-button fa fa-times-circle',
          'aria-hidden': true,
          onClick: handleClear,
          disabled,
        }),
      ]
    );
  }

  if (Children.count(children) === 2) {
    const value = Children.toArray(children)[1].props.children;
    return e(
      'div',
      {
        className: [
          'select',
          'select--value',
          disabled && 'select--disabled',
          className,
        ]
          .filter(Boolean)
          .join(' '),
        title: value,
      },
      e('span', { className: 'select__value' }, value)
    );
  }

  return e(
    'select',
    {
      name,
      className: ['select', className].filter(Boolean).join(' '),
      onChange: handleChange,
      disabled,
    },
    children
  );
};

export default Select;
