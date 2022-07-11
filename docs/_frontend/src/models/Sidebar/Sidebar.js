import Combobox from '../Combobox';
import Radio from '../Radio/Radio';
import RadioGroup from '../RadioGroup';
import styles from './Sidebar.module.css';

const Sidebar = ({ meta, params, onSubmit }) => {
  let tags = [];
  let predictedEntities = [];
  if (
    Array.isArray(meta?.aggregations?.tags) &&
    Array.isArray(meta?.aggregations?.predictedEntities)
  ) {
    ({ tags, predictedEntities } = meta.aggregations);
  }

  return (
    <div className={styles.root}>
      <div className={styles.control}>
        <RadioGroup
          value={params.type || ''}
          onChange={(value) => {
            onSubmit({ type: value });
          }}
        >
          <Radio value="">All</Radio>
          <Radio value="model">Models</Radio>
          <Radio value="pipeline">Pipelines</Radio>
        </RadioGroup>
      </div>

      <div className={styles.control}>
        <Combobox
          label="Assigned tags"
          items={tags}
          initialSelectedItems={params.tags || []}
          onChange={(values) => {
            onSubmit({ tags: values });
          }}
        />
      </div>

      <div className={styles.control}>
        <Combobox
          label="Entities"
          items={predictedEntities}
          initialSelectedItems={params.predicted_entities || []}
          onChange={(values) => {
            onSubmit({ predicted_entities: values });
          }}
        />
      </div>

      <div className={styles.control}>
        <RadioGroup
          value={params.sort || 'date'}
          onChange={(value) => {
            onSubmit({ sort: value });
          }}
        >
          <label>Sort By:</label>
          <div className={styles.radioContainers}>
            <Radio value="date" className={styles.radio}>
              Date
            </Radio>
            <Radio value="views" className={styles.radio}>
              Views
            </Radio>
            <Radio value="downloads" className={styles.radio}>
              Downloads
            </Radio>
          </div>
        </RadioGroup>
        <label>
          <input
            type="checkbox"
            name="recommended"
            checked={params.recommended}
            onChange={(e) => {
              onSubmit({ recommended: e.target.checked });
            }}
          />
          <span>Show recommended first</span>
        </label>
      </div>
    </div>
  );
};

export default Sidebar;
