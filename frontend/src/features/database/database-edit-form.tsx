import { Button, Card, Input, Select, Textarea } from "../../components/ui";

interface DatabaseEditFormProps {
  record: Record<string, unknown> | null;
  isSaving: boolean;
  onFieldChange: (field: string, value: unknown) => void;
  onSave: () => void;
}

function stringValue(value: unknown): string {
  if (Array.isArray(value)) {
    return value.join("、");
  }
  return value === null || value === undefined ? "" : String(value);
}

function triStateValue(value: unknown): string {
  if (value === true) {
    return "true";
  }
  if (value === false) {
    return "false";
  }
  return "";
}

function parseRiskFactors(value: string): string[] {
  return value
    .split(/[，,、;；\n]+/)
    .map((item) => item.trim())
    .filter(Boolean);
}

export function DatabaseEditForm({ record, isSaving, onFieldChange, onSave }: DatabaseEditFormProps) {
  return (
    <Card>
      <div className="database-section-heading">
        <h2>{"记录编辑"}</h2>
        <p className="clinical-copy clinical-copy-tight">
          {"基于当前虚拟病例库的 Excel 源数据进行单条写回。"}
        </p>
      </div>
      {record ? (
        <div className="database-edit-grid">
          <Input label="Patient ID" type="text" value={stringValue(record.patient_id)} readOnly />
          <Input
            label={"年龄"}
            aria-label={"年龄"}
            type="number"
            value={stringValue(record.age)}
            onChange={(event) => onFieldChange("age", event.target.value)}
          />
          <Input
            label={"性别"}
            type="text"
            value={stringValue(record.gender)}
            onChange={(event) => onFieldChange("gender", event.target.value)}
          />
          <Input
            label="ECOG"
            aria-label="ECOG"
            type="number"
            min={0}
            max={5}
            value={stringValue(record.ecog_score)}
            onChange={(event) => onFieldChange("ecog_score", event.target.value)}
          />
          <Input
            label={"肿瘤部位"}
            type="text"
            value={stringValue(record.tumor_location)}
            onChange={(event) => onFieldChange("tumor_location", event.target.value)}
          />
          <Input
            label={"病理类型"}
            type="text"
            value={stringValue(record.histology_type)}
            onChange={(event) => onFieldChange("histology_type", event.target.value)}
          />
          <Input
            label="MMR"
            type="text"
            value={stringValue(record.mmr_status)}
            onChange={(event) => onFieldChange("mmr_status", event.target.value)}
          />
          <label className="database-field database-field-span-2">
            <span className="database-field-label">{"主诉"}</span>
            <Textarea
              aria-label={"主诉"}
              rows={3}
              value={stringValue(record.chief_complaint)}
              onChange={(event) => onFieldChange("chief_complaint", event.target.value)}
            />
          </label>
          <Input
            label={"症状持续时间"}
            aria-label={"症状持续时间"}
            type="text"
            value={stringValue(record.symptom_duration)}
            onChange={(event) => onFieldChange("symptom_duration", event.target.value)}
          />
          <Select
            label={"家族史"}
            aria-label={"家族史"}
            value={triStateValue(record.family_history)}
            onChange={(event) =>
              onFieldChange("family_history", event.target.value === "" ? null : event.target.value === "true")
            }
          >
            <option value="">{"未填写"}</option>
            <option value="true">{"是"}</option>
            <option value="false">{"否"}</option>
          </Select>
          <label className="database-field database-field-span-2">
            <span className="database-field-label">{"家族史详情"}</span>
            <Textarea
              aria-label={"家族史详情"}
              rows={3}
              value={stringValue(record.family_history_details)}
              onChange={(event) => onFieldChange("family_history_details", event.target.value)}
            />
          </label>
          <Select
            label={"病理活检确认"}
            aria-label={"病理活检确认"}
            value={triStateValue(record.biopsy_confirmed)}
            onChange={(event) =>
              onFieldChange("biopsy_confirmed", event.target.value === "" ? null : event.target.value === "true")
            }
          >
            <option value="">{"未填写"}</option>
            <option value="true">{"是"}</option>
            <option value="false">{"否"}</option>
          </Select>
          <label className="database-field database-field-span-2">
            <span className="database-field-label">{"活检详情"}</span>
            <Textarea
              aria-label={"活检详情"}
              rows={3}
              value={stringValue(record.biopsy_details)}
              onChange={(event) => onFieldChange("biopsy_details", event.target.value)}
            />
          </label>
          <div className="database-field-span-2">
            <Input
              label={"危险因素"}
              aria-label={"危险因素"}
              type="text"
              value={stringValue(record.risk_factors)}
              onChange={(event) => onFieldChange("risk_factors", parseRiskFactors(event.target.value))}
            />
          </div>
        </div>
      ) : (
        <p className="clinical-copy">{"尚未选择可编辑的患者记录。"}</p>
      )}
      <div className="database-action-row">
        <Button type="button" disabled={!record || isSaving} onClick={onSave}>
          {isSaving ? "保存中..." : "保存记录"}
        </Button>
      </div>
    </Card>
  );
}
