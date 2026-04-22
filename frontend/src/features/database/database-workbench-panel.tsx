import type {
  DatabaseSearchRequest,
  DatabaseSearchResponse,
  DatabaseSortField,
  DatabaseStatsResponse,
  DatabaseWorkbenchMode,
} from "../../app/api/types";
import { DatabaseNaturalQueryBar } from "./database-natural-query-bar";
import { DatabaseResultsTable } from "./database-results-table";
import { numericSummary, topDistributionEntry } from "./use-database-workbench";

interface DatabaseWorkbenchPanelProps {
  title?: string;
  mode: DatabaseWorkbenchMode;
  naturalQuery: string;
  stats: DatabaseStatsResponse | null;
  searchRequest: DatabaseSearchRequest;
  searchResponse: DatabaseSearchResponse | null;
  selectedPatientId: number | null;
  isParsing: boolean;
  isSearching: boolean;
  isLoadingDetail: boolean;
  isBootstrapping: boolean;
  warnings: string[];
  unsupportedTerms: string[];
  error: string | null;
  onNaturalQueryChange: (value: string) => void;
  onNaturalQuerySubmit: () => void;
  onSelectPatient: (patientId: number) => void;
  onSortChange: (field: DatabaseSortField) => void;
  onPageChange: (page: number) => void;
  onClose?: () => void;
}

type DistributionKind = "gender" | "location" | "stage" | "mmr";

interface DistributionItem {
  label: string;
  count: number;
  share: number;
}

const MODE_LABELS: Record<DatabaseWorkbenchMode, string> = {
  stats: "\u5168\u5e93\u7edf\u8ba1",
  search: "\u7b5b\u9009\u68c0\u7d22",
  detail: "\u75c5\u4f8b\u8be6\u60c5",
  edit: "\u7f16\u8f91\u5199\u56de",
};

function normalizeDistributionLabel(kind: DistributionKind, label: string): string {
  const trimmed = label.trim();
  if (!trimmed) {
    return "-";
  }

  if (kind === "gender") {
    const lower = trimmed.toLowerCase();
    if (lower === "male" || lower === "m" || trimmed === "\u7537") {
      return "\u7537";
    }
    if (lower === "female" || lower === "f" || trimmed === "\u5973") {
      return "\u5973";
    }
  }

  return trimmed;
}

function buildDistributionItems(
  entries: Record<string, number> | undefined,
  kind: DistributionKind,
): DistributionItem[] {
  if (!entries) {
    return [];
  }

  const merged = new Map<string, number>();
  for (const [label, rawCount] of Object.entries(entries)) {
    if (!Number.isFinite(rawCount) || rawCount <= 0) {
      continue;
    }

    const normalizedLabel = normalizeDistributionLabel(kind, label);
    merged.set(normalizedLabel, (merged.get(normalizedLabel) ?? 0) + rawCount);
  }

  const total = Array.from(merged.values()).reduce((sum, count) => sum + count, 0);
  if (total <= 0) {
    return [];
  }

  return Array.from(merged.entries())
    .sort((left, right) => {
      if (right[1] !== left[1]) {
        return right[1] - left[1];
      }
      return left[0].localeCompare(right[0], "zh-CN");
    })
    .map(([label, count]) => ({
      label,
      count,
      share: (count / total) * 100,
    }));
}

function formatShare(share: number): string {
  if (!Number.isFinite(share)) {
    return "0%";
  }

  const rounded = Math.round(share * 10) / 10;
  if (Math.abs(rounded - Math.round(rounded)) < 0.05) {
    return `${Math.round(rounded)}%`;
  }
  return `${rounded.toFixed(1)}%`;
}

function renderDistributionCard(
  title: string,
  items: DistributionItem[],
  emptyText: string,
) {
  const headingCopy = items.length > 0
    ? "\u57fa\u4e8e\u5f53\u524d\u5168\u5e93\u6709\u6548\u75c5\u4f8b\u8ba1\u7b97\u5360\u6bd4\u3002"
    : "\u5f53\u524d\u6682\u65e0\u53ef\u7528\u7edf\u8ba1\u6837\u672c\u3002";

  return (
    <div className="workspace-card">
      <div className="database-section-heading">
        <h2>{title}</h2>
        <p className="workspace-copy workspace-copy-tight">{headingCopy}</p>
      </div>
      {items.length > 0 ? (
        <div className="database-distribution-list">
          {items.map((item) => (
            <div className="database-distribution-row" key={`${title}-${item.label}`}>
              <div className="database-distribution-row-head">
                <span>{item.label}</span>
                <span className="database-distribution-value">
                  {item.count} · {formatShare(item.share)}
                </span>
              </div>
              <div className="database-distribution-track" aria-hidden="true">
                <span
                  className="database-distribution-fill"
                  style={{ width: `${Math.min(100, Math.max(item.share, 8))}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      ) : (
        <p className="workspace-copy database-distribution-empty">{emptyText}</p>
      )}
    </div>
  );
}

export function DatabaseWorkbenchPanel({
  title = "\u6570\u636e\u5e93\u5de5\u4f5c\u53f0",
  mode,
  naturalQuery,
  stats,
  searchRequest,
  searchResponse,
  selectedPatientId,
  isParsing,
  isSearching,
  isLoadingDetail,
  isBootstrapping,
  warnings,
  unsupportedTerms,
  error,
  onNaturalQueryChange,
  onNaturalQuerySubmit,
  onSelectPatient,
  onSortChange,
  onPageChange,
  onClose,
}: DatabaseWorkbenchPanelProps) {
  const genderItems = buildDistributionItems(stats?.gender_distribution, "gender");
  const locationItems = buildDistributionItems(stats?.tumor_location_distribution, "location");
  const stageItems = buildDistributionItems(stats?.ct_stage_distribution, "stage");
  const mmrItems = buildDistributionItems(stats?.mmr_status_distribution, "mmr");

  return (
    <section className="workspace-panel-stack" data-testid="database-workbench-panel">
      <div className="workspace-card">
        <div className="database-section-heading database-section-heading-inline">
          <div>
            <h2>{title}</h2>
            <p className="workspace-copy workspace-copy-tight">
              {"\u8bc6\u522b\u5230\u6570\u636e\u5e93\u76f8\u5173\u610f\u56fe\u540e\uff0c\u8fd9\u91cc\u4f1a\u540c\u6b65\u5c55\u793a\u5168\u5e93\u7edf\u8ba1\u3001\u7b5b\u9009\u7ed3\u679c\u548c\u75c5\u4f8b\u5165\u53e3\u3002"}
            </p>
          </div>
          <div className="database-badge-row">
            <span className="workspace-stage-badge">{MODE_LABELS[mode]}</span>
            {onClose ? (
              <button
                type="button"
                className="workspace-secondary-button"
                onClick={onClose}
                aria-label={"\u6536\u8d77\u6570\u636e\u5e93\u5de5\u4f5c\u53f0"}
              >
                {"\u6536\u8d77"}
              </button>
            ) : null}
          </div>
        </div>
      </div>

      <DatabaseNaturalQueryBar
        value={naturalQuery}
        warnings={warnings}
        unsupportedTerms={unsupportedTerms}
        isParsing={isParsing}
        onChange={onNaturalQueryChange}
        onSubmit={onNaturalQuerySubmit}
      />

      {error ? (
        <div className="workspace-banner workspace-banner-error">
          <strong>{"\u6570\u636e\u5e93\u5de5\u4f5c\u53f0\u51fa\u9519"}</strong>
          <p className="workspace-copy workspace-copy-tight">{error}</p>
        </div>
      ) : null}

      {isBootstrapping ? <div className="workspace-banner">{"\u6b63\u5728\u540c\u6b65\u6570\u636e\u5e93\u5de5\u4f5c\u53f0..."}</div> : null}

      <div className="database-stat-grid">
        <div className="workspace-card">
          <h2>{"\u603b\u75c5\u4f8b\u6570"}</h2>
          <p className="workspace-metric">{stats?.total_cases ?? (isBootstrapping ? "..." : 0)}</p>
        </div>
        <div className="workspace-card">
          <h2>{"\u5e74\u9f84 (min/max/mean)"}</h2>
          <p className="workspace-metric database-metric-small">{numericSummary(stats?.age_statistics)}</p>
        </div>
        <div className="workspace-card">
          <h2>{"\u6700\u591a\u90e8\u4f4d"}</h2>
          <p className="workspace-metric database-metric-small">
            {stats ? topDistributionEntry(stats.tumor_location_distribution) : "..."}
          </p>
        </div>
        <div className="workspace-card">
          <h2>{"CEA (min/max/mean)"}</h2>
          <p className="workspace-metric database-metric-small">{numericSummary(stats?.cea_statistics)}</p>
        </div>
      </div>

      <div className="database-distribution-grid">
        {renderDistributionCard(
          "\u6027\u522b\u5206\u5e03",
          genderItems,
          "\u6682\u65e0\u53ef\u7528\u6027\u522b\u7edf\u8ba1\u3002",
        )}
        {renderDistributionCard(
          "\u90e8\u4f4d\u5206\u5e03",
          locationItems,
          "\u6682\u65e0\u53ef\u7528\u90e8\u4f4d\u7edf\u8ba1\u3002",
        )}
        {renderDistributionCard(
          "\u5206\u671f\u5206\u5e03",
          stageItems,
          "\u6682\u65e0\u53ef\u7528\u5206\u671f\u7edf\u8ba1\u3002",
        )}
        {renderDistributionCard(
          "MMR \u5206\u5e03",
          mmrItems,
          "\u6682\u65e0\u53ef\u7528 MMR \u7edf\u8ba1\u3002",
        )}
      </div>

      <DatabaseResultsTable
        items={searchResponse?.items ?? []}
        total={searchResponse?.total ?? 0}
        pagination={searchRequest.pagination}
        sort={searchRequest.sort}
        selectedPatientId={selectedPatientId}
        isSearching={isSearching || isBootstrapping}
        isLoadingDetail={isLoadingDetail}
        onSelectPatient={onSelectPatient}
        onSortChange={onSortChange}
        onPageChange={onPageChange}
      />
    </section>
  );
}
