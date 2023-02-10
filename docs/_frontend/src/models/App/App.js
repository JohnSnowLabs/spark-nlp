import React, { useState, useEffect } from 'react';
import Form from '../Form';
import ModelItem from '../ModelItem';
import ModelItemList from '../ModelItemList';
import { SEARCH_ORIGIN, toSearchString } from '../common';

const fromSearchString = () => {
  const params = {};
  const searchParams = new URLSearchParams(window.location.search);
  [
    'q',
    'edition',
    'language',
    'task',
    'supported',
    'type',
    'tags',
    'sort',
    'predicted_entities',
    'page',
  ].forEach((key) => {
    if (!searchParams.has(key)) {
      return;
    }
    switch (key) {
      case 'tags':
      case 'predicted_entities':
        params[key] = searchParams.getAll(key);
        break;

      case 'supported':
        if (searchParams.get(key) === '1') {
          params[key] = true;
        }
        break;

      default:
        params[key] = searchParams.get(key);
        break;
    }
  });
  params['recommended'] =
    !searchParams.has('recommended') || searchParams.get('recommended') !== '0';
  return params;
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
          document.getElementById('app').scrollIntoView({ behavior: 'smooth' });
        }
      });
  }, [params]);

  const dispatch = (event) => {
    if (state.value === 'loading') {
      return;
    }
    let nextParams;
    switch (event._type) {
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

const App = () => {
  const [state, send] = useFilterQuery();

  const handleSubmit = (params) => {
    send({ _type: 'SUBMIT', ...params, page: undefined });
  };

  const handlePageChange = (page) => {
    send({ _type: 'CHANGE_PAGE', page });
  };

  const handleSupportedToggle = (supported) => {
    send({ _type: 'SUBMIT', page: undefined, supported });
  };

  let result;
  switch (true) {
    case Boolean(state.context.data):
      {
        let children;
        if (state.context.data.length > 0) {
          children = state.context.data.map((item) => (
            <ModelItem key={item.url} {...item} />
          ));
        } else {
          children = (
            <div className="model-items__no-results">
              Sorry, but there are no results. Try other filtering options.
            </div>
          );
        }
        result = (
          <ModelItemList
            meta={state.context.meta}
            params={state.context.params}
            onSubmit={handleSubmit}
            onPageChange={handlePageChange}
            onSupportedToggle={handleSupportedToggle}
          >
            {children}
          </ModelItemList>
        );
      }
      break;

    case Boolean(state.context.error):
      break;

    default:
      break;
  }

  return (
    <>
      <Form
        onSubmit={handleSubmit}
        isLoading={state.value === 'loading'}
        params={state.context.params}
        meta={state.context.meta}
      />
      {result}
    </>
  );
};

export default App;
