import type { DatabaseFilters } from "../../app/api/types";
import { Button, Card, Input, Select } from "../../components/ui";

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
    <Card>
      <div className="database-section-heading">
        <h2>{"结构化筛选"}</h2>
        <p className="clinical-copy clinical-copy-tight">
          {"自然语言解析目前仍只覆盖旧筛选项；家族史、活检确认和 ECOG 区间请在这里手动设置。"}
        </p>
      </div>
      <div className="database-filter-grid">
        <Input
          label="Patient ID"
          type="number"
          value={numberValue(filters.patient_id)}
          onChange={(event) =>
            onFiltersChange({
              ...filters,
              patient_id: event.target.value ? Number(event.target.value) : null,
            })
          }
        />
        <Input
          label={"年龄下限"}
          type="number"
          value={numberValue(filters.age_min)}
          onChange={(event) =>
            onFiltersChange({
              ...filters,
              age_min: event.target.value ? Number(event.target.value) : null,
            })
          }
        />
        <Input
          label={"年龄上限"}
          type="number"
          value={numberValue(filters.age_max)}
          onChange={(event) =>
            onFiltersChange({
              ...filters,
              age_max: event.target.value ? Number(event.target.value) : null,
            })
          }
        />
        <Select
          label={"肿瘤部位"}
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
        </Select>
        <Select
          label="MMR"
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
        </Select>
        <Select
          label={"家族史"}
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
        </Select>
        <Select
          label={"活检确认"}
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
        </Select>
        <Select
          label={"ECOG 下限"}
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
        </Select>
        <Select
          label={"ECOG 上限"}
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
        </Select>
      </div>
      <div className="database-action-row">
        <Button type="button" onClick={onApply} disabled={isSearching}>
          {isSearching ? "检索中..." : "应用筛选"}
        </Button>
        <Button type="button" variant="secondary" onClick={onReset} disabled={isSearching}>
          {"重置"}
        </Button>
      </div>
    </Card>
  );
}
