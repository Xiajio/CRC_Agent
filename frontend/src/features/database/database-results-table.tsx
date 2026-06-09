import type { DatabaseCaseRow, DatabasePagination, DatabaseSort } from "../../app/api/types";
import { Button, Card } from "../../components/ui";

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
    <Card>
      <div className="database-section-heading database-section-heading-inline">
        <h2>{"病例列表"}</h2>
        <span className="clinical-stage-badge">{`${total} 条`}</span>
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
                      <Button
                        type="button"
                        className="database-table-button"
                        size="sm"
                        variant="ghost"
                        onClick={() => onSelectPatient(patientId)}
                        disabled={isSearching || isLoadingDetail}
                      >
                        {`查看 ${patientId}`}
                      </Button>
                    </td>
                  </tr>
                );
              })
            ) : (
              <tr>
                <td colSpan={8}>
                  <p className="clinical-copy">{"当前筛选条件下暂无病例。"}</p>
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      <div className="database-pagination">
        <Button
          type="button"
          variant="secondary"
          onClick={() => onPageChange(Math.max(1, pagination.page - 1))}
          disabled={pagination.page <= 1 || isSearching}
        >
          {"上一页"}
        </Button>
        <span className="clinical-copy clinical-copy-tight">{`${pagination.page} / ${totalPages}`}</span>
        <Button
          type="button"
          variant="secondary"
          onClick={() => onPageChange(Math.min(totalPages, pagination.page + 1))}
          disabled={pagination.page >= totalPages || isSearching}
        >
          {"下一页"}
        </Button>
      </div>
    </Card>
  );
}
