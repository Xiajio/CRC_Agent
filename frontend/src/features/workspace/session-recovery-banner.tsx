interface SessionRecoveryBannerProps {
  message: string;
  onDismiss: () => void;
}

const containerStyle: React.CSSProperties = {
  background: "#FFF7E5",
  border: "1px solid #F0C36D",
  color: "#7A5300",
  padding: "10px 14px",
  borderRadius: 8,
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
  gap: 12,
  margin: "8px 16px 0",
  fontSize: 13,
  lineHeight: 1.5,
};

const buttonStyle: React.CSSProperties = {
  background: "transparent",
  border: "none",
  color: "#7A5300",
  cursor: "pointer",
  fontSize: 16,
  lineHeight: 1,
  padding: 4,
};

export function SessionRecoveryBanner({ message, onDismiss }: SessionRecoveryBannerProps) {
  return (
    <div
      role="status"
      data-testid="session-recovery-banner"
      aria-live="polite"
      style={containerStyle}
    >
      <span>{message}</span>
      <button
        type="button"
        onClick={onDismiss}
        aria-label="关闭恢复提示"
        style={buttonStyle}
      >
        ×
      </button>
    </div>
  );
}
