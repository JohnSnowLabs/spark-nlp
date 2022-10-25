import React from 'react';
import ModelItemTag from '../ModelItemTag';
import { productDisplayName, addNamingConventions } from './utils';
import './ModelItem.css';

const ModelItem = ({
  title,
  url,
  task,
  language,
  edition,
  date,
  supported,
  recommended,
  deprecated,
  highlight,
}) => {
  let body;
  if (highlight && highlight.body && highlight.body[0]) {
    body = highlight.body[0];
  }

  const getDisplayedDate = () => {
    const [year, month] = date.split('-');
    return month + '.' + year;
  };

  let label;
  if (deprecated) {
    label = <div className="model-item__deprecated">Deprecated</div>;
  } else if (supported) {
    label = <div className="model-item__supported">Supported</div>;
  }

  return (
    <div className="cell cell--12 cell--md-6 cell--lg-4">
      <div className="model-item">
        {label}
        {recommended && (
          <span class="fa fa-star model-item__recommended"></span>
        )}
        <div className="model-item__header">
          <a href={url} className="model-item__title">
            {addNamingConventions(title)}
          </a>
        </div>
        <div className="model-item__content">
          {body && (
            <div
              className="model-item__highlight"
              dangerouslySetInnerHTML={{ __html: body }}
            />
          )}
          <ModelItemTag
            icon="calendar-alt"
            name="Date"
            value={getDisplayedDate()}
          />
          <ModelItemTag
            icon="edit"
            name="task"
            value={Array.isArray(task) ? task.join(', ') : task}
          />
          <ModelItemTag icon="flag" name="Language" value={language} />
          <ModelItemTag
            icon="clone"
            name="Edition"
            value={productDisplayName(edition)}
          />
        </div>
      </div>
    </div>
  );
};

export default ModelItem;
