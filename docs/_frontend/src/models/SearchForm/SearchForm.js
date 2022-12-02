import { createElement as e, useState, useEffect } from 'react';
import Spinner from '../Spinner';
import './SearchForm.css';

const SearchForm = ({ onSubmit, isLoading, params }) => {
  const [value, setValue] = useState('');
  useEffect(() => {
    setValue((params ? params.q : undefined) || '');
  }, [params]);
  return e(
    'form',
    {
      className: 'search-form models-hero__group',
      onSubmit: (e) => {
        e.preventDefault();
        onSubmit({ q: value });
      },
    },
    e('input', {
      type: 'text',
      name: 'q',
      value,
      onChange: (e) => setValue(e.target.value),
      className: 'search-form__input',
      placeholder: 'Search models and pipelines',
    }),
    !!(params && params.q) &&
      e('button', {
        key: 'button',
        type: 'button',
        className: 'search-form__input-clear-button fa fa-times-circle',
        onClick: () => {
          onSubmit({ q: undefined });
        },
      }),

    isLoading
      ? e('span', { className: 'search-form__button' }, e(Spinner))
      : e('button', {
          className: 'fa fa-search search-form__button',
          type: 'submit',
        })
  );
};

export default SearchForm;
