import Select from '../Select';
import './FilterForm.css';

const { createElement: e } = React;

const FilterForm = ({ onSubmit, isLoading, meta, params }) => {
  let tasks = [];
  let languages = [];
  let editions = [];
  if (meta && meta.aggregations) {
    ({ tasks, languages, editions } = meta.aggregations);
  }
  editions.sort((a, b) => {
    if (a.indexOf('Healthcare') !== -1 && b.indexOf('Healthcare') === -1) {
      return 1;
    }
    return 0;
  });

  let task;
  let language;
  let edition;
  if (params) {
    ({ task, language, edition } = params);
  }

  const languageOptions = [
    [e('option', { key: 0, value: '' }, 'All Languages')],
  ];
  languages.forEach(({ code, name }) => {
    languageOptions.push(e('option', { key: code, value: code }, name));
  });

  return e(
    'form',
    {
      className: 'filter-form models-hero__group',
      onSubmit: (e) => {
        e.preventDefault();
      },
    },
    [
      e('span', { key: 0, className: 'filter-form__group' }, [
        e('span', { key: 0, className: 'filter-form__text' }, 'Show'),
        e(
          Select,
          {
            key: 1,
            name: 'task',
            value: task,
            className: 'filter-form__select filter-form__select--task',
            disabled: isLoading,
            onChange: (e) => {
              onSubmit({ task: e.target.value });
            },
          },
          tasks.reduce(
            (acc, task) => {
              acc.push(e('option', { key: task, value: task }, task));
              return acc;
            },
            [[e('option', { key: 0, value: '' }, 'All')]]
          )
        ),
      ]),
      e('span', { key: 1, className: 'filter-form__group' }, [
        e(
          'span',
          { key: 0, className: 'filter-form__text' },
          'models & pipelines in'
        ),
        e(
          Select,
          {
            key: 1,
            name: 'language',
            value: language,
            renderValue: (code) => {
              const item = languages.find((v) => v.code === code);
              return item ? item.name : code;
            },
            className: 'filter-form__select filter-form__select--language',
            disabled: isLoading,
            onChange: (e) => {
              onSubmit({ language: e.target.value });
            },
          },
          languageOptions
        ),
      ]),
      e('span', { key: 2, className: 'filter-form__group' }, [
        e('span', { key: 0, className: 'filter-form__text' }, 'for'),
        e(
          Select,
          {
            key: 1,
            name: 'edition',
            value: edition,
            className: 'filter-form__select filter-form__select--edition',
            disabled: isLoading,
            onChange: (e) => {
              onSubmit({ edition: e.target.value });
            },
          },
          editions
            .map((edition) =>
              e('option', { key: edition, value: edition }, edition)
            )
            .reduce(
              (acc, item) => {
                if (
                  item.key !== 'Spark NLP' &&
                  item.key !== 'Spark NLP for Healthcare'
                ) {
                  acc.push(item);
                }
                return acc;
              },
              [
                e('option', { key: 'all', value: '' }, 'All versions'),
                e(
                  'option',
                  {
                    key: 'All Spark NLP versions',
                    value: 'Spark NLP',
                  },
                  'All Spark NLP versions'
                ),
                e(
                  'option',
                  {
                    key: 'All Spark NLP for Healthcare versions',
                    value: 'Spark NLP for Healthcare',
                  },
                  'All Spark NLP for Healthcare versions'
                ),
              ]
            )
        ),
      ]),
    ]
  );
};

export default FilterForm;
