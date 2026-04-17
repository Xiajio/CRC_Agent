import type { DatabaseFilters } from "../../app/api/types";

interface DatabaseFiltersPanelProps {
  filters: DatabaseFilters;
  isSearching: boolean;
  onFiltersChange: (nextFilters: DatabaseFilters) => void;
  onApply: () => void;
  onReset: () => void;
}

const TUMOR_LOCATION_OPTIONS = [
  { value: "", label: "全部部位" },
  { value: "直肠", label: "直肠" },
  { value: "横结肠", label: "横结肠" },
  { value: "升结肠", label: "升结肠" },
  { value: "降结肠", label: "降结肠" },
  { value: "乙状结肠", label: "乙状结肠" },
  { value: "盲肠", label: "盲肠" },
];

const MMR_OPTIONS = [
  { value: "", label: "全部 MMR" },
  { value: "pMMR_MSS", label: "pMMR / MSS" },
  { value: "dMMR_MSI_H", label: "dMMR / MSI-H" },
];

const TRI_STATE_OPTIONS = [
  { value: "", label: "未填写" },
  { value: "true", label: "是" },
  { value: "false", label: "否" },
];

const ECOG_OPTIONS = ["", "0", "1", "2", "3", "4", "5"];

function numberValue(value: number | null | undefined): string {
  return value === null || value === undefined ? "" : String(value);
}

function triStateValue(value: boolean | null | undefined): string {
  if (value === true) {
    return "true";
  }
  if (value === false) {
    return "false";
  }
  return "";
}

function readTriState(value: string): boolean | null {
  if (value === "true") {
    return true;
  }
  if (value === "false") {
    return false;
  }
  return null;
}

export function DatabaseFiltersPanel({
  filters,
  isSearching,
  onFiltersChange,
  onApply,
  onReset,
}: DatabaseFiltersPanelProps) {
  return (
    <div className="workspace-card">
      <div className="database-section-heading">
        <h2>{"结构化筛选"}</h2>
        <p className="workspace-copy workspace-copy-tight">
          {"自然语言解析目前仍只覆盖旧筛选项；家族史、活检确认和 ECOG 区间请在这里手动设置。"}
        </p>
      </div>
      <div className="database-filter-grid">
        <label className="database-field">
          <span className="database-field-label">Patient ID</span>
          <input
            className="database-input"
            type="number"
            value={numberValue(filters.patient_id)}
            onChange={(event) =>
              onFiltersChange({
                ...filters,
                patient_id: event.target.value ? Number(event.target.value) : null,
              })
            }
          />
        </label>
        <label className="database-field">
          <span className="database-field-label">{"年龄下限"}</span>
          <input
            className="database-input"
            type="number"
            value={numberValue(filters.age_min)}
            onChange={(event) =>
              onFiltersChange({
                ...filters,
                age_min: event.target.value ? Number(event.target.value) : null,
              })
            }
          />
        </label>
        <label className="database-field">
          <span className="database-field-label">{"年龄上限"}</span>
          <input
            className="database-input"
            type="number"
            value={numberValue(filters.age_max)}
            onChange={(event) =>
              onFiltersChange({
                ...filters,
                age_max: event.target.value ? Number(event.target.value) : null,
              })
            }
          />
        </label>
        <label className="database-field">
          <span className="database-field-label">{"肿瘤部位"}</span>
          <select
            className="database-select"
            value={filters.tumor_location[0] ?? ""}
            onChange={(event) =>
              onFiltersChange({
                ...filters,
                tumor_location: event.target.value ? [event.target.value] : [],
              })
            }
          >
            {TUMOR_LOCATION_OPTIONS.map((option) => (
              <option key={option.value || "all"} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
        <label className="database-field">
          <span className="database-field-label">MMR</span>
          <select
            className="database-select"
            value={filters.mmr_status[0] ?? ""}
            onChange={(event) =>
              onFiltersChange({
                ...filters,
                mmr_status: event.target.value ? [event.target.value] : [],
              })
            }
          >
            {MMR_OPTIONS.map((option) => (
              <option key={option.value || "all"} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
        <label className="database-field">
          <span className="database-field-label">{"家族史"}</span>
          <select
            className="database-select"
            aria-label={"家族史筛选"}
            value={triStateValue(filters.family_history)}
            onChange={(event) =>
              onFiltersChange({
                ...filters,
                family_history: readTriState(event.target.value),
              })
            }
          >
            {TRI_STATE_OPTIONS.map((option) => (
              <option key={option.value || "empty"} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
        <label className="database-field">
          <span className="database-field-label">{"活检确认"}</span>
          <select
            className="database-select"
            aria-label={"活检确认筛选"}
            value={triStateValue(filters.biopsy_confirmed)}
            onChange={(event) =>
              onFiltersChange({
                ...filters,
                biopsy_confirmed: readTriState(event.target.value),
              })
            }
          >
            {TRI_STATE_OPTIONS.map((option) => (
              <option key={option.value || "empty"} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
        <label className="database-field">
          <span className="database-field-label">{"ECOG 下限"}</span>
          <select
            className="database-select"
            aria-label={"ECOG 下限"}
            value={numberValue(filters.ecog_min)}
            onChange={(event) =>
              onFiltersChange({
                ...filters,
                ecog_min: event.target.value ? Number(event.target.value) : null,
              })
            }
          >
            {ECOG_OPTIONS.map((option) => (
              <option key={option || "empty"} value={option}>
                {option === "" ? "未填写" : option}
              </option>
            ))}
          </select>
        </label>
        <label className="database-field">
          <span className="database-field-label">{"ECOG 上限"}</span>
          <select
            className="database-select"
            aria-label={"ECOG 上限"}
            value={numberValue(filters.ecog_max)}
            onChange={(event) =>
              onFiltersChange({
                ...filters,
                ecog_max: event.target.value ? Number(event.target.value) : null,
              })
            }
          >
            {ECOG_OPTIONS.map((option) => (
              <option key={option || "empty"} value={option}>
                {option === "" ? "未填写" : option}
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className="database-action-row">
        <button type="button" className="workspace-button" onClick={onApply} disabled={isSearching}>
          {isSearching ? "检索中..." : "应用筛选"}
        </button>
        <button type="button" className="workspace-secondary-button" onClick={onReset} disabled={isSearching}>
          {"重置"}
        </button>
      </div>
    </div>
  );
}
