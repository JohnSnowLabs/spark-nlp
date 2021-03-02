(function () {
  const { createElement: e, useState, useEffect, Fragment } = React;

  const SEARCH_ORIGIN = 'https://search.modelshub.johnsnowlabs.com';

  const tasks = [
    'Named Entity Recognition',
    'Text Classification',
    'Sentiment Analysis',
    'Translation',
    'Question Answering',
    'Summarization',
    'Sentence Detection',
    'Embeddings',
    'Language Detection',
    'Stop Words Removal',
    'Word Segmentation',
    'Part of Speech Tagging',
    'Lemmatization',
    'Relation Extraction',
    'Spell Checking',
    'Assertion Status',
    'Entity Resolution',
    'De-identification',
  ];

  const languages = {
    en: 'English',
    xx: 'Multilingual',
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

  const languageGroups = {
    aav: 'Austro-Asiatic languages',
    afa: 'Afro-Asiatic languages',
    alg: 'Algonquian languages',
    alv: 'Atlantic-Congo languages',
    apa: 'Apache languages',
    aqa: 'Alacalufan languages',
    aql: 'Algic languages',
    art: 'Artificial languages',
    ath: 'Athapascan languages',
    auf: 'Arauan languages',
    aus: 'Australian languages',
    awd: 'Arawakan languages',
    azc: 'Uto-Aztecan languages',
    bad: 'Banda languages',
    bai: 'Bamileke languages',
    bat: 'Baltic languages',
    ber: 'Berber languages',
    bih: 'Bihari languages',
    bnt: 'Bantu languages',
    btk: 'Batak languages',
    cai: 'Central American Indian languages',
    cau: 'Caucasian languages',
    cba: 'Chibchan languages',
    ccn: 'North Caucasian languages',
    ccs: 'South Caucasian languages',
    cdc: 'Chadic languages',
    cdd: 'Caddoan languages',
    cel: 'Celtic languages',
    cmc: 'Chamic languages',
    cpe: 'English based creoles and pidgins',
    cpf: 'French-based creoles and pidgins',
    cpp: 'Portuguese-based creoles and pidgins',
    crp: 'Creoles and pidgins',
    csu: 'Central Sudanic languages',
    cus: 'Cushitic languages',
    day: 'Land Dayak languages',
    dmn: 'Mande languages',
    dra: 'Dravidian languages',
    egx: 'Egyptian languages',
    esx: 'Eskimo-Aleut languages',
    euq: 'Basque (family)',
    fiu: 'Finno-Ugrian languages',
    fox: 'Formosan languages',
    gem: 'Germanic languages',
    gme: 'East Germanic languages',
    gmq: 'North Germanic languages',
    gmw: 'West Germanic languages',
    grk: 'Greek languages',
    him: 'Himachali languages, Western Pahari languages',
    hmx: 'Hmong-Mien languages',
    hok: 'Hokan languages',
    hyx: 'Armenian (family)',
    iir: 'Indo-Iranian languages',
    ijo: 'Ijo languages',
    inc: 'Indic languages',
    ine: 'Indo-European languages',
    ira: 'Iranian languages',
    iro: 'Iroquoian languages',
    itc: 'Italic languages',
    jpx: 'Japanese (family)',
    kar: 'Karen languages',
    kdo: 'Kordofanian languages',
    khi: 'Khoisan languages',
    kro: 'Kru languages',
    map: 'Austronesian languages',
    mkh: 'Mon-Khmer languages',
    mno: 'Manobo languages',
    mun: 'Munda languages',
    myn: 'Mayan languages',
    nah: 'Nahuatl languages',
    nai: 'North American Indian',
    ngf: 'Trans-New Guinea languages',
    nic: 'Niger-Kordofanian languages',
    nub: 'Nubian languages',
    omq: 'Oto-Manguean languages',
    omv: 'Omotic languages',
    oto: 'Otomian languages',
    paa: 'Papuan languages',
    phi: 'Philippine languages',
    plf: 'Central Malayo-Polynesian languages',
    poz: 'Malayo-Polynesian languages',
    pqe: 'Eastern Malayo-Polynesian languages',
    pqw: 'Western Malayo-Polynesian languages',
    pra: 'Prakrit languages',
    qwe: 'Quechuan (family)',
    roa: 'Romance languages',
    sai: 'South American Indian languages',
    sal: 'Salishan languages',
    sdv: 'Eastern Sudanic languages',
    sem: 'Semitic languages',
    sgn: 'Sign Languages',
    sio: 'Siouan languages',
    sit: 'Sino-Tibetan languages',
    sla: 'Slavic languages',
    smi: 'Sami languages',
    son: 'Songhai languages',
    sqj: 'Albanian languages',
    ssa: 'Nilo-Saharan languages',
    syd: 'Samoyedic languages',
    tai: 'Tai languages',
    tbq: 'Tibeto-Burman languages',
    trk: 'Turkic languages',
    tup: 'Tupi languages',
    tut: 'Altaic languages',
    tuw: 'Tungus languages',
    urj: 'Uralic languages',
    wak: 'Wakashan languages',
    wen: 'Sorbian languages',
    xgn: 'Mongolian languages',
    xnd: 'Na-Dene languages',
    ypk: 'Yupik languages',
    zhx: 'Chinese (family)',
    zle: 'East Slavic languages',
    zls: 'South Slavic languages',
    zlw: 'West Slavic languages',
    znd: 'Zande languages',
  };

  const editions = [
    'Spark NLP 2.7',
    'Spark NLP 2.6',
    'Spark NLP 2.5',
    'Spark NLP 2.4',
    'Spark NLP for Healthcare 2.7',
    'Spark NLP for Healthcare 2.6',
    'Spark NLP for Healthcare 2.5',
    'Spark NLP for Healthcare 2.4',
  ];

  const useFilterQuery = () => {
    const [state, setState] = useState({
      value: 'idle',
      context: {},
    });

    const [params, setParams] = useState({});

    useEffect(() => {
      setState({
        ...state,
        value: 'loading',
        context: { ...state.context, form: params.type },
      });
      const mapping = {
        edition: 'edition_short',
      };
      const searchParams = Object.keys(params).reduce((acc, k) => {
        if (params[k] && k !== 'type') {
          acc.append(mapping[k] || k, params[k]);
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
        case 'FILTER':
        case 'SEARCH':
          setParams(event);
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

  const Spinner = () => e('i', { className: 'fas fa-circle-notch fa-spin' });

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

  const ModelItem = ({
    title,
    url,
    task,
    language,
    edition,
    date,
    highlight,
  }) => {
    const getDisplayedLanguage = () => {
      if (!language) {
        return null;
      }
      switch (language.length) {
        case 2:
          return languages[language] || language;

        case 3:
          return languageGroups[language] || language;

        default:
          return language;
      }
    };

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
        e(
          'div',
          { key: 'header', className: 'model-item__header' },
          e('a', { href: url, className: 'model-item__title', title }, title)
        ),
        e('div', { key: 'content', className: 'model-item__content' }, [
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
            value: getDisplayedLanguage(),
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

  const FilterForm = ({
    onSubmit,
    onTaskChange,
    isLoading,
    isHealthcareOnly,
  }) => {
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
              onChange: onTaskChange,
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
            'select',
            {
              key: 1,
              name: 'language',
              className:
                'select filter-form__select filter-form__select--language',
            },
            Object.keys(languages)
              .reduce(
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
                [[e('option', { key: 0, value: '' }, 'All Languages')]]
              )
              .concat(
                [
                  e(
                    'option',
                    { key: 'separator', disabled: true },
                    '──────────'
                  ),
                ],
                Object.keys(languageGroups).map((code) => {
                  return e(
                    'option',
                    { key: code, value: code },
                    languageGroups[code]
                  );
                })
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
            editions
              .filter((edition) => {
                if (isHealthcareOnly) {
                  return edition.indexOf('for Healthcare') !== -1;
                }
                return true;
              })
              .map((edition) =>
                e('option', { key: edition, value: edition }, edition)
              )
              .reduce(
                (acc, item) => {
                  acc.push(item);
                  return acc;
                },
                [
                  e(
                    'option',
                    { key: 'all', value: '' },
                    'All Spark NLP versions'
                  ),
                ]
              )
          ),
          e(
            'button',
            {
              key: 2,
              type: 'submit',
              className: 'button filter-form__button',
            },
            isLoading ? e(Spinner) : 'Go'
          ),
        ]),
      ]
    );
  };

  const SearchForm = ({ onSubmit, isLoading }) => {
    return e(
      'form',
      {
        className: 'search-form',
        onSubmit,
      },
      e('input', {
        type: 'text',
        name: 'q',
        className: 'search-form__input',
        placeholder: 'Search models and pipelines',
      }),
      e(
        'button',
        {
          className: 'button search-form__button',
          type: 'submit',
        },
        isLoading ? e(Spinner) : 'Search'
      )
    );
  };

  const SearchAndUpload = ({ onSubmit, isLoading }) => {
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
        e(SearchForm, { onSubmit, isLoading })
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
    const [isHealthcareOnly, setIsHealthcareOnly] = useState(false);

    const handleFilterSubmit = (e) => {
      e.preventDefault();
      const {
        target: {
          task: { value: task },
          language: { value: language },
          edition: { value: edition },
        },
      } = e;
      send({ type: 'FILTER', task, language, edition });
    };

    const handleSearchSubmit = (e) => {
      e.preventDefault();
      const {
        target: {
          q: { value: q },
        },
      } = e;
      send({ type: 'SEARCH', q });
    };

    const handleTaskChange = (e) => {
      setIsHealthcareOnly(
        ['Assertion Status', 'Entity Resolution', 'De-identification'].indexOf(
          e.target.value
        ) !== -1
      );
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
        onTaskChange: handleTaskChange,
        isLoading: state.value === 'loading' && state.context.form === 'FILTER',
        isHealthcareOnly,
      }),
      e(SearchAndUpload, {
        key: 1,
        onSubmit: handleSearchSubmit,
        isLoading: state.value === 'loading' && state.context.form === 'SEARCH',
      }),
      result,
    ]);
  };

  ReactDOM.render(e(App), document.getElementById('app'));
})();
