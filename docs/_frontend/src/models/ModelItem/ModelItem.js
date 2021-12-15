import ModelItemTag from '../ModelItemTag';
import './ModelItem.css';

const { createElement: e } = React;

const ModelItem = ({
  title,
  url,
  task,
  language,
  edition,
  date,
  supported,
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
    label = e(
      'div',
      { key: 'deprecated', className: 'model-item__deprecated' },
      'Deprecated'
    );
  } else if (supported) {
    label = e(
      'div',
      { key: 'supported', className: 'model-item__supported' },
      'Supported'
    );
  }

  return e(
    'div',
    { className: 'cell cell--12 cell--md-6 cell--lg-4' },
    e('div', { className: 'model-item' }, [
      label,
      e(
        'div',
        { key: 'header', className: 'model-item__header' },
        e('a', { href: url, className: 'model-item__title', title }, title)
      ),
      e('div', { key: 'content', className: 'model-item__content' }, [
        body &&
          e('div', {
            key: 'body',
            className: 'model-item__highlight',
            dangerouslySetInnerHTML: { __html: body },
          }),
        e(ModelItemTag, {
          key: 'date',
          icon: 'calendar-alt',
          name: 'Date',
          value: getDisplayedDate(),
        }),
        e(ModelItemTag, {
          key: 'task',
          icon: 'edit',
          name: 'Task',
          value: Array.isArray(task) ? task.join(', ') : task,
        }),
        e(ModelItemTag, {
          key: 'language',
          icon: 'flag',
          name: 'Language',
          value: language,
        }),
        e(ModelItemTag, {
          key: 'edition',
          icon: 'clone',
          name: 'Edition',
          value: edition,
        }),
      ]),
    ])
  );
};

export default ModelItem;
