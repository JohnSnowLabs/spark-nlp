import Select from '../Select';
import './FilterForm.css';

const { createElement: e } = React;

const compareEditions = (a, b) => {
  const getPriority = (edition) => {
    const products = ['Spark NLP', 'Spark NLP for Healthcare', 'Spark OCR'];
    const index = products.indexOf(edition);
    if (index !== -1) {
      return index - products.length;
    }
    if (edition.includes('Spark NLP for Healthcare')) {
      return 1;
    }
    if (edition.includes('Spark NLP')) {
      return 0;
    }
    return 2;
  };
  return getPriority(a) - getPriority(b);
};
const FilterForm = ({ onSubmit, isLoading, meta, params }) => {
  let tasks = [];
  let languages = [];
  let editions = [];
  if (meta && meta.aggregations) {
    ({ tasks, languages, editions } = meta.aggregations);
  }
  editions.sort(compareEditions);

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
            .map((edition) => {
              if (
                edition == 'Spark NLP' ||
                edition == 'Spark NLP for Healthcare' ||
                edition == 'Spark OCR'
              ) {
                let new_edition = `All ${edition} versions`;
                return e(
                  'option',
                  { key: new_edition, value: edition },
                  new_edition
                );
              } else {
                return e('option', { key: edition, value: edition }, edition);
              }
            })
            .reduce(
              (acc, item) => {
                acc.push(item);
                return acc;
              },
              [e('option', { key: 'all', value: '' }, 'All versions')]
            )
        ),
      ]),
    ]
  );
};

export default FilterForm;
