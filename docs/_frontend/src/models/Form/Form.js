import SearchForm from '../SearchForm';
import FilterForm from '../FilterForm';

const { createElement: e } = React;

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

export default Form;
