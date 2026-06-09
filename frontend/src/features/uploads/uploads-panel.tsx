import { useRef } from "react";

import { Button, Card } from "../../components/ui";

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
    <Card variant="clinical-panel" data-testid="uploads-panel">
      <h2>资料上传</h2>
      <p className="clinical-copy">已恢复资料：{assetIds.length}</p>
      {assetIds.length > 0 ? (
        <ul className="clinical-list" data-testid="uploaded-assets-list">
          {assetIds.map((assetId) => (
            <li key={assetId} className="clinical-list-item" data-testid={`uploaded-asset-${assetId}`}>
              <strong data-testid="uploaded-asset-id">{assetId}</strong>
            </li>
          ))}
        </ul>
      ) : (
        <p className="clinical-copy">当前暂无已上传资料</p>
      )}
      {statusMessage ? (
        <p className="clinical-copy" data-testid="upload-status">
          {statusMessage}
        </p>
      ) : null}
      <input
        ref={inputRef}
        data-testid="upload-input"
        className="clinical-upload-input"
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
      <Button
        type="button"
        variant="secondary"
        disabled={disabled}
        onClick={() => inputRef.current?.click()}
      >
        上传资料
      </Button>
    </Card>
  );
}
