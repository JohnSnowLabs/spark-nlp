(function () {
  const { createElement: e, useState, useEffect, Fragment } = React;

  const SEARCH_ORIGIN = 'https://search.modelshub.johnsnowlabs.com';

  const tasks = [
    'Named Entity Recognition',
    'Text Classification',
    'Sentiment Analysis',
    'Assertion Status',
    'Entity Resolution',
    'Translation',
    'Question Answering',
    'Summarization',
    'Sentence Detection',
    'Embeddings',
    'Language Detection',
    'Deidentification',
    'Stop Words',
    'Word Segmentation',
    'POS',
    'Lemmatization',
    'Relation Extraction',
    'Pipeline Healthcare',
    'Pipeline Translate',
    'Pipeline Onto NER',
    'Pipeline Language Detection',
  ];

  const languages = {
    aa: 'Afar',
    ab: 'Abkhaz',
    ae: 'Avestan',
    af: 'Afrikaans',
    ak: 'Akan',
    am: 'Amharic',
    an: 'Aragonese',
    ar: 'Arabic',
    as: 'Assamese',
    av: 'Avaric',
    ay: 'Aymara',
    az: 'Azerbaijani',
    ba: 'Bashkir',
    be: 'Belarusian',
    bg: 'Bulgarian',
    bh: 'Bihari',
    bi: 'Bislama',
    bm: 'Bambara',
    bn: 'Bengali, Bangla',
    bo: 'Tibetan Standard, Tibetan, Central',
    br: 'Breton',
    bs: 'Bosnian',
    ca: 'Catalan',
    ce: 'Chechen',
    ch: 'Chamorro',
    co: 'Corsican',
    cr: 'Cree',
    cs: 'Czech',
    cu: 'Old Church Slavonic, Church Slavonic, Old Bulgarian',
    cv: 'Chuvash',
    cy: 'Welsh',
    da: 'Danish',
    de: 'German',
    dv: 'Divehi, Dhivehi, Maldivian',
    dz: 'Dzongkha',
    ee: 'Ewe',
    el: 'Greek (modern)',
    en: 'English',
    eo: 'Esperanto',
    es: 'Spanish',
    et: 'Estonian',
    eu: 'Basque',
    fa: 'Persian (Farsi)',
    ff: 'Fula, Fulah, Pulaar, Pular',
    fi: 'Finnish',
    fj: 'Fijian',
    fo: 'Faroese',
    fr: 'French',
    fy: 'Western Frisian',
    ga: 'Irish',
    gd: 'Scottish Gaelic, Gaelic',
    gl: 'Galician',
    gn: 'Guaraní',
    gu: 'Gujarati',
    gv: 'Manx',
    ha: 'Hausa',
    he: 'Hebrew (modern)',
    hi: 'Hindi',
    ho: 'Hiri Motu',
    hr: 'Croatian',
    ht: 'Haitian, Haitian Creole',
    hu: 'Hungarian',
    hy: 'Armenian',
    hz: 'Herero',
    ia: 'Interlingua',
    id: 'Indonesian',
    ie: 'Interlingue',
    ig: 'Igbo',
    ii: 'Nuosu',
    ik: 'Inupiaq',
    io: 'Ido',
    is: 'Icelandic',
    it: 'Italian',
    iu: 'Inuktitut',
    ja: 'Japanese',
    jv: 'Javanese',
    ka: 'Georgian',
    kg: 'Kongo',
    ki: 'Kikuyu, Gikuyu',
    kj: 'Kwanyama, Kuanyama',
    kk: 'Kazakh',
    kl: 'Kalaallisut, Greenlandic',
    km: 'Khmer',
    kn: 'Kannada',
    ko: 'Korean',
    kr: 'Kanuri',
    ks: 'Kashmiri',
    ku: 'Kurdish',
    kv: 'Komi',
    kw: 'Cornish',
    ky: 'Kyrgyz',
    la: 'Latin',
    lb: 'Luxembourgish, Letzeburgesch',
    lg: 'Ganda',
    li: 'Limburgish, Limburgan, Limburger',
    ln: 'Lingala',
    lo: 'Lao',
    lt: 'Lithuanian',
    lu: 'Luba-Katanga',
    lv: 'Latvian',
    mg: 'Malagasy',
    mh: 'Marshallese',
    mi: 'Māori',
    mk: 'Macedonian',
    ml: 'Malayalam',
    mn: 'Mongolian',
    mr: 'Marathi (Marāṭhī)',
    ms: 'Malay',
    mt: 'Maltese',
    my: 'Burmese',
    na: 'Nauruan',
    nb: 'Norwegian Bokmål',
    nd: 'Northern Ndebele',
    ne: 'Nepali',
    ng: 'Ndonga',
    nl: 'Dutch',
    nn: 'Norwegian Nynorsk',
    no: 'Norwegian',
    nr: 'Southern Ndebele',
    nv: 'Navajo, Navaho',
    ny: 'Chichewa, Chewa, Nyanja',
    oc: 'Occitan',
    oj: 'Ojibwe, Ojibwa',
    om: 'Oromo',
    or: 'Oriya',
    os: 'Ossetian, Ossetic',
    pa: '(Eastern) Punjabi',
    pi: 'Pāli',
    pl: 'Polish',
    ps: 'Pashto, Pushto',
    pt: 'Portuguese',
    qu: 'Quechua',
    rm: 'Romansh',
    rn: 'Kirundi',
    ro: 'Romanian',
    ru: 'Russian',
    rw: 'Kinyarwanda',
    sa: 'Sanskrit (Saṁskṛta)',
    sc: 'Sardinian',
    sd: 'Sindhi',
    se: 'Northern Sami',
    sg: 'Sango',
    si: 'Sinhalese, Sinhala',
    sk: 'Slovak',
    sl: 'Slovene',
    sm: 'Samoan',
    sn: 'Shona',
    so: 'Somali',
    sq: 'Albanian',
    sr: 'Serbian',
    ss: 'Swati',
    st: 'Southern Sotho',
    su: 'Sundanese',
    sv: 'Swedish',
    sw: 'Swahili',
    ta: 'Tamil',
    te: 'Telugu',
    tg: 'Tajik',
    th: 'Thai',
    ti: 'Tigrinya',
    tk: 'Turkmen',
    tl: 'Tagalog',
    tn: 'Tswana',
    to: 'Tonga (Tonga Islands)',
    tr: 'Turkish',
    ts: 'Tsonga',
    tt: 'Tatar',
    tw: 'Twi',
    ty: 'Tahitian',
    ug: 'Uyghur',
    uk: 'Ukrainian',
    ur: 'Urdu',
    uz: 'Uzbek',
    ve: 'Venda',
    vi: 'Vietnamese',
    vo: 'Volapük',
    wa: 'Walloon',
    wo: 'Wolof',
    xh: 'Xhosa',
    yi: 'Yiddish',
    yo: 'Yoruba',
    za: 'Zhuang, Chuang',
    zh: 'Chinese',
    zu: 'Zulu',
  };

  const editions = [
    'Spark NLP 2.4.0',
    'Spark NLP 2.4.1',
    'Spark NLP 2.4.2',
    'Spark NLP 2.4.3',
    'Spark NLP 2.4.4',
    'Spark NLP 2.4.5',
    'Spark NLP 2.5.0',
    'Spark NLP 2.5.1',
    'Spark NLP 2.5.2',
    'Spark NLP 2.5.3',
    'Spark NLP 2.5.4',
    'Spark NLP 2.5.5',
    'Spark NLP 2.6.0',
    'Spark NLP 2.6.1',
    'Spark NLP 2.6.2',
    'Spark NLP 2.6.3',
    'Spark NLP 2.6.4',
    'Spark NLP 2.6.5',
    'Spark NLP 2.7.0',
    'Spark NLP for Healthcare 2.4.0',
    'Spark NLP for Healthcare 2.4.1',
    'Spark NLP for Healthcare 2.4.2',
    'Spark NLP for Healthcare 2.4.5',
    'Spark NLP for Healthcare 2.4.6',
    'Spark NLP for Healthcare 2.5.0',
    'Spark NLP for Healthcare 2.5.2',
    'Spark NLP for Healthcare 2.5.3',
    'Spark NLP for Healthcare 2.5.5',
    'Spark NLP for Healthcare 2.6.0',
    'Spark NLP for Healthcare 2.6.2',
    'Spark NLP for Healthcare 2.7.0',
    'Spark NLP for Healthcare 2.7.1',
    'Spark NLP for Healthcare 2.7.2',
  ];

  const useFilterQuery = () => {
    const [state, setState] = useState({
      value: 'idle',
      context: {},
    });

    const [params, setParams] = useState({});

    useEffect(() => {
      setState({ ...state, value: 'loading' });
      const searchParams = Object.keys(params).reduce((acc, k) => {
        if (params[k] && k !== 'type') {
          acc.append(k, params[k]);
        }
        return acc;
      }, new URLSearchParams());
      fetch(SEARCH_ORIGIN + '/?' + searchParams)
        .then((res) => {
          if (res.ok) {
            return res.json();
          }
          throw new Error('Search is not available at the moment.');
        })
        .then(({ data, meta }) => {
          setState({
            value: 'success',
            context: { data, meta },
          });
        })
        .catch((e) => {
          setState({
            value: 'failure',
            context: { error: e.message },
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
      switch (event.type) {
        case 'SUBMIT':
          {
            const { task, language, edition } = event;
            setParams({ task, language, edition });
          }
          break;

        case 'CHANGE_PAGE':
          {
            const { page } = event;
            setParams({ ...params, page });
          }
          break;
      }
    };
    return [state, dispatch];
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

  const ModelItemList = ({ meta, children, onPageChange }) => {
    return ReactDOM.createPortal(
      e(Fragment, null, [
        e('div', { key: 0, className: 'grid--container model-items' }, [
          e('div', { key: 0, className: 'grid' }, children),
          e(Pagination, { key: 1, ...meta, onChange: onPageChange }),
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

  const ModelItem = ({ title, url, task, language, edition }) => {
    return e(
      'div',
      { className: 'cell cell--12 cell--md-6 cell--lg-4' },
      e('div', { className: 'model-item' }, [
        e(
          'div',
          { key: 'header', className: 'model-item__header' },
          e('a', { href: url, className: 'model-item__title', title }, title)
        ),
        e('div', { key: 'content', className: 'model-item__content' }, [
          e(ModelItemTag, { key: 0, icon: 'edit', name: 'Task', value: task }),
          e(ModelItemTag, {
            key: 1,
            icon: 'flag',
            name: 'Language',
            value: language,
          }),
          e(ModelItemTag, {
            key: 2,
            icon: 'clone',
            name: 'Edition',
            value: edition,
          }),
        ]),
      ])
    );
  };

  const FilterForm = ({ onSubmit, isLoading }) => {
    return e(
      'form',
      { className: 'filter-form models-hero__group', onSubmit },
      [
        e('span', { key: 0, className: 'filter-form__group' }, [
          e('span', { key: 0, className: 'filter-form__text' }, 'Show'),
          e(
            'select',
            {
              key: 1,
              name: 'task',
              className: 'select filter-form__select filter-form__select--task',
            },
            tasks.reduce(
              (acc, task) => {
                acc.push(e('option', { key: task, value: task }, task));
                return acc;
              },
              [[e('option', { key: 0, value: '' }, 'Task')]]
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
            'select',
            {
              key: 1,
              name: 'language',
              className:
                'select filter-form__select filter-form__select--language',
            },
            Object.keys(languages).reduce(
              (acc, language) => {
                acc.push(
                  e(
                    'option',
                    { key: language, value: language },
                    languages[language]
                  )
                );
                return acc;
              },
              [[e('option', { key: 0, value: '' }, 'Language')]]
            )
          ),
        ]),
        e('span', { key: 2, className: 'filter-form__group' }, [
          e('span', { key: 0, className: 'filter-form__text' }, 'for'),
          e(
            'select',
            {
              key: 1,
              name: 'edition',
              className:
                'select filter-form__select filter-form__select--edition',
            },
            editions.reduce(
              (acc, edition) => {
                acc.push(
                  e('option', { key: edition, value: edition }, edition)
                );
                return acc;
              },
              [[e('option', { key: 0, value: '' }, 'Spark NLP edition')]]
            )
          ),
          e(
            'button',
            {
              key: 2,
              type: 'submit',
              className: 'button filter-form__button',
            },
            isLoading ? '...' : 'Go'
          ),
        ]),
      ]
    );
  };

  const SearchAndUpload = () => {
    return e(
      'div',
      {
        className: 'models-hero__group models-hero__group--search-and-upload',
      },
      e(
        'div',
        {
          className: 'models-hero__item models-hero__item--search',
        },
        e(
          'form',
          {
            className: 'search-form',
          },
          e('input', {
            type: 'text',
            className: 'search-form__input',
            placeholder: 'Search models and pipelines',
          }),
          e(
            'button',
            {
              className: 'button search-form__button',
            },
            'Search'
          )
        )
      ),
      e(
        'div',
        {
          className: 'models-hero__item models-hero__item--upload',
        },
        e(
          'a',
          {
            href: 'https://modelshub.johnsnowlabs.com/',
            className: 'button models-hero__upload-button',
          },
          e('i', {
            className: 'fa fa-upload',
          }),
          '\xA0 Upload Your Model'
        )
      )
    );
  };

  const App = () => {
    const [state, send] = useFilterQuery();

    const handleFilterSubmit = (e) => {
      e.preventDefault();
      const {
        target: {
          task: { value: task },
          language: { value: language },
          edition: { value: edition },
        },
      } = e;
      send({ type: 'SUBMIT', task, language, edition });
    };

    const handlePageChange = (page) => {
      send({ type: 'CHANGE_PAGE', page });
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
            onPageChange: handlePageChange,
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
      e(FilterForm, {
        key: 0,
        onSubmit: handleFilterSubmit,
        isLoading: state.value === 'loading',
      }),
      e(SearchAndUpload, { key: 1 }),
      result,
    ]);
  };

  ReactDOM.render(e(App), document.getElementById('app'));
})();
