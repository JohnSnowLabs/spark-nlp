import Pagination from '../Pagination';
import './ModelItemList.css';

const { createElement: e, Fragment } = React;

const formatNumber = (value) => {
  return String(value)
    .split('')
    .reverse()
    .map((v, i) => (i > 0 && i % 3 === 0 ? v + ',' : v))
    .reverse()
    .join('');
};

const ModelItemList = ({
  meta,
  params,
  children,
  onPageChange,
  onSupportedToggle,
}) => {
  const handleSupportedChange = (e) => {
    const flag = e.target.checked;
    onSupportedToggle(flag);
  };

  return ReactDOM.createPortal(
    e(Fragment, null, [
      e('div', { key: 'items', className: 'grid--container model-items' }, [
        e(
          'div',
          {
            key: 'header',
            className: 'model-items__header',
          },
          e(
            'div',
            { className: 'model-items__header-total-results' },
            formatNumber(meta.totalItems) + ' Models & Pipelines Results:'
          ),
          e('label', { className: 'model-items__header-supported' }, [
            e('input', {
              type: 'checkbox',
              name: 'supported',
              checked: params.supported,
              onChange: handleSupportedChange,
            }),
            e('span', {}, 'Supported models only'),
          ])
        ),
        e('div', { key: 'grid', className: 'grid' }, children),
        e(Pagination, { key: 'pagination', ...meta, onChange: onPageChange }),
      ]),
    ]),

    document.getElementById('results')
  );
};

export default ModelItemList;
