import { createElement as e } from 'react';

const ModelItemTag = ({ icon, name, value }) => {
  return e('div', { className: 'model-item__tag' }, [
    e(
      'span',
      { key: 'icon', className: 'model-item__tag-icon' },
      e('i', { className: 'far fa-' + icon })
    ),
    e('span', { key: 'name', className: 'model-item__tag-name' }, name + ':'),
    e('strong', { key: 'value', className: 'model-item__tag-value' }, value),
  ]);
};

export default ModelItemTag;
