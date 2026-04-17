import { useRef } from "react";

type UploadsPanelProps = {
  uploadedAssets: Record<string, unknown>;
  disabled?: boolean;
  statusMessage?: string | null;
  onUpload: (file: File) => void;
};

export function UploadsPanel({
  uploadedAssets,
  disabled = false,
  statusMessage,
  onUpload,
}: UploadsPanelProps) {
  const assetIds = Object.keys(uploadedAssets);
  const inputRef = useRef<HTMLInputElement | null>(null);

  return (
    <div className="workspace-card" data-testid="uploads-panel">
      <h2>资料上传</h2>
      <p className="workspace-copy">已恢复资料：{assetIds.length}</p>
      {assetIds.length > 0 ? (
        <ul className="workspace-list" data-testid="uploaded-assets-list">
          {assetIds.map((assetId) => (
            <li key={assetId} className="workspace-list-item" data-testid={`uploaded-asset-${assetId}`}>
              <strong data-testid="uploaded-asset-id">{assetId}</strong>
            </li>
          ))}
        </ul>
      ) : (
        <p className="workspace-copy">当前暂无已上传资料</p>
      )}
      {statusMessage ? (
        <p className="workspace-copy" data-testid="upload-status">
          {statusMessage}
        </p>
      ) : null}
      <input
        ref={inputRef}
        data-testid="upload-input"
        className="workspace-upload-input"
        type="file"
        disabled={disabled}
        onChange={(event) => {
          const nextFile = event.target.files?.[0];
          if (nextFile) {
            onUpload(nextFile);
            event.target.value = "";
          }
        }}
      />
      <button
        type="button"
        className="workspace-button"
        disabled={disabled}
        onClick={() => inputRef.current?.click()}
      >
        上传资料
      </button>
    </div>
  );
}
