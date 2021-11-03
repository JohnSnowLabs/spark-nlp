(function () {
  const { createElement: e, useState, useEffect, Fragment, Children } = React;

  const SEARCH_ORIGIN = 'https://search.modelshub.johnsnowlabs.com';

  const toSearchString = (params) => {
    let isEmpty = true;
    const searchParams = Object.keys(params).reduce((acc, k) => {
      if (params[k] && k !== 'type') {
        let value = params[k];
        if (k === 'supported') {
          value = Number(value);
        }
        acc.append(k, value);
        isEmpty = false;
      }
      return acc;
    }, new URLSearchParams());

    return isEmpty ? '' : '?' + searchParams;
  };

  const fromSearchString = () => {
    const params = Object.fromEntries(
      new URLSearchParams(window.location.search)
    );
    if ('supported' in params && params.supported === '1') {
      params.supported = true;
    }
    return params;
  };

  const formatNumber = (value) => {
    return String(value)
      .split('')
      .reverse()
      .map((v, i) => (i > 0 && i % 3 === 0 ? v + ',' : v))
      .reverse()
      .join('');
  };

  const useFilterQuery = () => {
    const [state, setState] = useState({
      value: 'idle',
      context: {},
    });

    const [params, setParams] = useState(fromSearchString());

    useEffect(() => {
      window.onpopstate = (e) => {
        setParams(
          Object.fromEntries(new URLSearchParams(window.location.search))
        );
      };
      return () => {
        window.onpopstate = null;
      };
    }, []);

    useEffect(() => {
      setState({
        ...state,
        value: 'loading',
        context: { ...state.context, params },
      });

      fetch(SEARCH_ORIGIN + '/' + toSearchString(params))
        .then((res) => {
          if (res.ok) {
            return res.json();
          }
          throw new Error('Search is not available at the moment.');
        })
        .then(({ data, meta }) => {
          setState({
            value: 'success',
            context: { data, meta, params },
          });
        })
        .catch((e) => {
          setState({
            value: 'failure',
            context: { error: e.message, params },
          });
        })
        .finally(() => {
          if (params.page) {
            document
              .getElementById('app')
              .scrollIntoView({ behavior: 'smooth' });
          }
        });
    }, [params]);

    const dispatch = (event) => {
      if (state.value === 'loading') {
        return;
      }
      let nextParams;
      switch (event.type) {
        case 'SUBMIT':
          nextParams = { ...params, ...event };
          break;

        case 'CHANGE_PAGE':
          {
            const { page } = event;
            nextParams = { ...params, page };
          }
          break;
      }
      window.history.pushState(
        nextParams,
        window.title,
        '/models' + toSearchString(nextParams)
      );
      setParams(nextParams);
    };
    return [state, dispatch];
  };

  const Spinner = () => e('i', { className: 'fas fa-circle-notch fa-spin' });

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

  const ModelItem = ({
    title,
    url,
    task,
    language,
    edition,
    date,
    supported,
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

    return e(
      'div',
      { className: 'cell cell--12 cell--md-6 cell--lg-4' },
      e('div', { className: 'model-item' }, [
        supported &&
          e(
            'div',
            {
              key: 'supported',
              className: 'model-item__supported',
            },
            'Supported'
          ),
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

  const Form = ({ onSubmit, isLoading, params, meta }) => {
    return [
      e(SearchForm, {
        key: 'search',
        onSubmit,
        isLoading,
        params,
      }),
      e(FilterForm, {
        key: 'filter',
        onSubmit,
        isLoading,
        meta,
        params,
      }),
    ];
  };

  const App = () => {
    const [state, send] = useFilterQuery();

    const handleSubmit = (params) => {
      send({ type: 'SUBMIT', ...params, page: undefined });
    };

    const handlePageChange = (page) => {
      send({ type: 'CHANGE_PAGE', page });
    };

    const handleSupportedToggle = (supported) => {
      send({ type: 'SUBMIT', page: undefined, supported });
    };

    let result;
    switch (true) {
      case Boolean(state.context.data):
        {
          let children;
          if (state.context.data.length > 0) {
            children = state.context.data.map((item) =>
              e(ModelItem, { key: item.url, ...item })
            );
          } else {
            children = e(
              'div',
              { className: 'model-items__no-results' },
              'Sorry, but there are no results. Try other filtering options.'
            );
          }
          result = e(ModelItemList, {
            key: 'model-items',
            meta: state.context.meta,
            params: state.context.params,
            onPageChange: handlePageChange,
            onSupportedToggle: handleSupportedToggle,
            children,
          });
        }
        break;

      case Boolean(state.context.error):
        break;

      default:
        break;
    }

    return e(React.Fragment, null, [
      e(Form, {
        key: 0,
        onSubmit: handleSubmit,
        isLoading: state.value === 'loading',
        params: state.context.params,
        meta: state.context.meta,
      }),
      result,
    ]);
  };

  ReactDOM.render(e(App), document.getElementById('app'));
})();
