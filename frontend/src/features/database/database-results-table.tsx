import type { DatabaseCaseRow, DatabasePagination, DatabaseSort } from "../../app/api/types";

interface DatabaseResultsTableProps {
  items: DatabaseCaseRow[];
  total: number;
  pagination: DatabasePagination;
  sort: DatabaseSort;
  selectedPatientId: number | null;
  isSearching: boolean;
  isLoadingDetail: boolean;
  onSelectPatient: (patientId: number) => void;
  onSortChange: (field: DatabaseSort["field"]) => void;
  onPageChange: (page: number) => void;
}

function sortLabel(sort: DatabaseSort, field: DatabaseSort["field"]): string {
  if (sort.field !== field) {
    return "";
  }
  return sort.direction === "asc" ? " ↑" : " ↓";
}

export function DatabaseResultsTable({
  items,
  total,
  pagination,
  sort,
  selectedPatientId,
  isSearching,
  isLoadingDetail,
  onSelectPatient,
  onSortChange,
  onPageChange,
}: DatabaseResultsTableProps) {
  const totalPages = Math.max(1, Math.ceil(total / pagination.page_size));

  return (
    <div className="workspace-card">
      <div className="database-section-heading database-section-heading-inline">
        <h2>{"病例列表"}</h2>
        <span className="workspace-stage-badge">{`${total} 条`}</span>
      </div>
      <div className="database-table-scroll">
        <table className="database-table">
          <thead>
            <tr>
              <th>
                <button type="button" className="database-table-sort" onClick={() => onSortChange("patient_id")}>
                  {`ID${sortLabel(sort, "patient_id")}`}
                </button>
              </th>
              <th>
                <button type="button" className="database-table-sort" onClick={() => onSortChange("age")}>
                  {`年龄${sortLabel(sort, "age")}`}
                </button>
              </th>
              <th>{"性别"}</th>
              <th>
                <button type="button" className="database-table-sort" onClick={() => onSortChange("ecog_score")}>
                  {`ECOG${sortLabel(sort, "ecog_score")}`}
                </button>
              </th>
              <th>{"部位"}</th>
              <th>{"分期"}</th>
              <th>MMR</th>
              <th>{"操作"}</th>
            </tr>
          </thead>
          <tbody>
            {items.length > 0 ? (
              items.map((item) => {
                const patientId = Number(item.patient_id);
                const selected = selectedPatientId === patientId;
                return (
                  <tr key={patientId} data-selected={selected ? "true" : "false"}>
                    <td>{patientId}</td>
                    <td>{item.age ?? "-"}</td>
                    <td>{item.gender ?? "-"}</td>
                    <td>{item.ecog_score ?? "-"}</td>
                    <td>{item.tumor_location ?? "-"}</td>
                    <td>{item.clinical_stage ?? "-"}</td>
                    <td>{item.mmr_status ?? "-"}</td>
                    <td>
                      <button
                        type="button"
                        className="workspace-secondary-button database-table-button"
                        onClick={() => onSelectPatient(patientId)}
                        disabled={isSearching || isLoadingDetail}
                      >
                        {`查看 ${patientId}`}
                      </button>
                    </td>
                  </tr>
                );
              })
            ) : (
              <tr>
                <td colSpan={8}>
                  <p className="workspace-copy">{"当前筛选条件下暂无病例。"}</p>
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      <div className="database-pagination">
        <button
          type="button"
          className="workspace-secondary-button"
          onClick={() => onPageChange(Math.max(1, pagination.page - 1))}
          disabled={pagination.page <= 1 || isSearching}
        >
          {"上一页"}
        </button>
        <span className="workspace-copy workspace-copy-tight">{`${pagination.page} / ${totalPages}`}</span>
        <button
          type="button"
          className="workspace-secondary-button"
          onClick={() => onPageChange(Math.min(totalPages, pagination.page + 1))}
          disabled={pagination.page >= totalPages || isSearching}
        >
          {"下一页"}
        </button>
      </div>
    </div>
  );
}
