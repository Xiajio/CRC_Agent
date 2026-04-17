interface DatabaseNaturalQueryBarProps {
  value: string;
  warnings: string[];
  unsupportedTerms: string[];
  isParsing: boolean;
  onChange: (value: string) => void;
  onSubmit: () => void;
}

export function DatabaseNaturalQueryBar({
  value,
  warnings,
  unsupportedTerms,
  isParsing,
  onChange,
  onSubmit,
}: DatabaseNaturalQueryBarProps) {
  return (
    <div className="workspace-card">
      <div className="database-section-heading">
        <h2>{"\u81ea\u7136\u8bed\u8a00\u67e5\u8be2"}</h2>
        <p className="workspace-copy workspace-copy-tight">
          {
            "\u8ba9\u5927\u6a21\u578b\u5148\u89e3\u6790\u68c0\u7d22\u610f\u56fe\uff0c\u518d\u540c\u6b65\u5230\u4e0b\u65b9\u7ed3\u6784\u5316\u7b5b\u9009\u3002"
          }
        </p>
      </div>
      <label className="database-field">
        <span className="database-field-label">{"\u81ea\u7136\u8bed\u8a00\u67e5\u8be2"}</span>
        <input
          className="database-input"
          type="text"
          value={value}
          onChange={(event) => onChange(event.target.value)}
          placeholder={"\u4f8b\u5982\uff1a\u5e2e\u6211\u627e\u51fa 30-40 \u5c81\u3001\u6709\u809d\u8f6c\u79fb\u7684\u60a3\u8005"}
        />
      </label>
      <div className="database-action-row">
        <button
          type="button"
          className="workspace-button"
          onClick={onSubmit}
          disabled={isParsing || !value.trim()}
        >
          {isParsing ? "\u89e3\u6790\u4e2d..." : "\u89e3\u6790\u67e5\u8be2"}
        </button>
      </div>
      {warnings.length > 0 ? (
        <div className="database-feedback-list" role="status">
          {warnings.map((warning) => (
            <p key={warning} className="workspace-copy workspace-copy-tight">
              {warning}
            </p>
          ))}
        </div>
      ) : null}
      {unsupportedTerms.length > 0 ? (
        <p className="workspace-copy workspace-copy-tight">
          {`\u672a\u652f\u6301\u7684\u6761\u4ef6\uff1a${unsupportedTerms.join("\u3001")}`}
        </p>
      ) : null}
    </div>
  );
}
