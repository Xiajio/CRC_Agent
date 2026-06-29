import type { PatientCareCardsResponse } from "../../app/api/types";
import { Card } from "../../components/ui";

type PatientCareCardsProps = {
  cards: PatientCareCardsResponse | null;
  isLoading: boolean;
};

const EMPTY_CARDS: PatientCareCardsResponse = {
  focusMetrics: [],
  periodicChecks: [],
  dailyActions: [],
};

function CareGroup({ title, items }: { title: string; items: string[] }) {
  return (
    <section className="clinical-care-card">
      <strong>{title}</strong>
      {items.length > 0 ? (
        <ul className="clinical-list">
          {items.map((item) => (
            <li key={item} className="clinical-list-item">
              {item}
            </li>
          ))}
        </ul>
      ) : (
        <p className="clinical-copy clinical-copy-tight">暂无可展示内容</p>
      )}
    </section>
  );
}

export function PatientCareCards({ cards, isLoading }: PatientCareCardsProps) {
  const display = cards ?? EMPTY_CARDS;
  return (
    <Card as="section" variant="clinical-panel">
      <h2>个人随访提醒</h2>
      {isLoading ? <p className="clinical-copy">正在加载随访提醒...</p> : null}
      {!isLoading ? (
        <div className="clinical-care-card-grid">
          <CareGroup title="最近需要留意的信号" items={display.focusMetrics} />
          <CareGroup title="可安排的检查事项" items={display.periodicChecks} />
          <CareGroup title="居家记录与行动" items={display.dailyActions} />
        </div>
      ) : null}
    </Card>
  );
}
