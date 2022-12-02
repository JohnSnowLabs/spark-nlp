import React from 'react';
import { createPortal } from 'react-dom';
import Pagination from '../Pagination';
import Sidebar from '../Sidebar';
import './ModelItemList.css';

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
  onSubmit,
  onPageChange,
  onSupportedToggle,
}) => {
  const handleSupportedChange = (e) => {
    const flag = e.target.checked;
    onSupportedToggle(flag);
  };

  return createPortal(
    <div key="items" className="grid--container model-items">
      <div className="model-items__sidebar-wrapper">
        <div className="model-items__sidebar">
          <Sidebar params={params} meta={meta} onSubmit={onSubmit} />
        </div>
      </div>
      <div className="model-items__content-wrapper">
        <div className="model-items__header">
          <div className="model-items__header-total-results">
            {formatNumber(meta.totalItems) + ' Models & Pipelines Results:'}
          </div>
          <label className="model-items__header-supported">
            <input
              type="checkbox"
              name="supported"
              checked={params.supported}
              onChange={handleSupportedChange}
            />
            <span>Supported models only</span>
          </label>
        </div>
        <div className="grid">{children}</div>
        <Pagination {...meta} onChange={onPageChange} />
      </div>
    </div>,
    document.getElementById('results')
  );
};

export default ModelItemList;
