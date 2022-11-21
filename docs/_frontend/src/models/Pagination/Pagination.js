import { createElement as e } from 'react';
import './Pagination.css';

const Pagination = ({ page, totalPages, onChange }) => {
  if (totalPages <= 1) {
    return null;
  }
  return e('div', { className: 'pagination' }, [
    e(
      'a',
      {
        key: 'first',
        className: 'pagination__button',
        disabled: page <= 1,
        onClick: () => {
          onChange(1);
        },
      },
      'first'
    ),
    e(
      'a',
      {
        key: 'prev',
        className: 'pagination__button',
        disabled: page <= 1,
        onClick: () => {
          onChange(page - 1);
        },
      },
      'previous'
    ),
    e(
      'span',
      { key: 'text', className: 'pagination__text' },
      page + ' page of ' + totalPages
    ),
    e(
      'a',
      {
        key: 'next',
        className: 'pagination__button',
        disabled: page >= totalPages,
        onClick: () => {
          onChange(page + 1);
        },
      },
      'next'
    ),
    e(
      'a',
      {
        key: 'last',
        className: 'pagination__button',
        disabled: page >= totalPages,
        onClick: () => {
          onChange(totalPages);
        },
      },
      'last'
    ),
  ]);
};

export default Pagination;
