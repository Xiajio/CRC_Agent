import { useEffect, useRef, useState } from "react";
import { Check, ChevronDown, ShieldCheck, Stethoscope, UserRound } from "lucide-react";

import { classNames } from "../../components/ui";

export type WorkspaceSurface = "patient" | "doctor" | "agent-admin";

type WorkspaceSurfaceMeta = {
  label: string;
  description: string;
  icon: typeof UserRound;
};

const SURFACE_META: Record<WorkspaceSurface, WorkspaceSurfaceMeta> = {
  patient: {
    label: "患者",
    description: "患者端问诊与资料",
    icon: UserRound,
  },
  doctor: {
    label: "医生",
    description: "医生端会诊与数据库",
    icon: Stethoscope,
  },
  "agent-admin": {
    label: "后台",
    description: "智能体运行观测",
    icon: ShieldCheck,
  },
};

export function surfaceLabel(surface: WorkspaceSurface): string {
  return SURFACE_META[surface].label;
}

type WorkspaceSurfaceSwitcherProps = {
  activeSurface: WorkspaceSurface;
  surfaces?: WorkspaceSurface[];
  onSelect: (surface: WorkspaceSurface) => void;
};

export function WorkspaceSurfaceSwitcher({
  activeSurface,
  surfaces = ["patient", "doctor", "agent-admin"],
  onSelect,
}: WorkspaceSurfaceSwitcherProps) {
  const [isOpen, setIsOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement | null>(null);
  const activeMeta = SURFACE_META[activeSurface];
  const ActiveIcon = activeMeta.icon;

  useEffect(() => {
    if (!isOpen) {
      return;
    }

    function handleDocumentPointerDown(event: MouseEvent) {
      if (event.target instanceof Node && !rootRef.current?.contains(event.target)) {
        setIsOpen(false);
      }
    }

    function handleDocumentKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setIsOpen(false);
      }
    }

    document.addEventListener("mousedown", handleDocumentPointerDown);
    document.addEventListener("keydown", handleDocumentKeyDown);
    return () => {
      document.removeEventListener("mousedown", handleDocumentPointerDown);
      document.removeEventListener("keydown", handleDocumentKeyDown);
    };
  }, [isOpen]);

  function handleSelect(surface: WorkspaceSurface) {
    setIsOpen(false);
    if (surface !== activeSurface) {
      onSelect(surface);
    }
  }

  return (
    <div ref={rootRef} className="clinical-surface-switcher">
      <button
        type="button"
        className="clinical-surface-trigger ui-profile-switch clinical-profile-switch"
        aria-label={`切换工作台，当前为${activeMeta.label}`}
        aria-haspopup="menu"
        aria-expanded={isOpen}
        onClick={() => setIsOpen((current) => !current)}
      >
        <span className="ui-profile-avatar clinical-avatar clinical-surface-avatar" aria-hidden="true">
          <ActiveIcon size={16} strokeWidth={2} />
        </span>
        <span className="ui-profile-label clinical-doctor-name">{activeMeta.label}</span>
        <ChevronDown className="clinical-surface-chevron" size={15} strokeWidth={2} aria-hidden="true" />
      </button>
      {isOpen ? (
        <div className="clinical-surface-menu" role="menu" aria-label="工作台切换">
          {surfaces.map((surface) => {
            const meta = SURFACE_META[surface];
            const Icon = meta.icon;
            const isActive = surface === activeSurface;
            return (
              <button
                key={surface}
                type="button"
                role="menuitem"
                className={classNames([
                  "clinical-surface-menu-item",
                  isActive && "clinical-surface-menu-item-active",
                ])}
                aria-current={isActive ? "page" : undefined}
                onClick={() => handleSelect(surface)}
              >
                <span className="clinical-surface-menu-icon" aria-hidden="true">
                  <Icon size={16} strokeWidth={2} />
                </span>
                <span className="clinical-surface-menu-copy">
                  <strong>{meta.label}</strong>
                  <small>{meta.description}</small>
                </span>
                {isActive ? (
                  <span className="clinical-surface-menu-check" aria-hidden="true">
                    <Check size={15} strokeWidth={2} />
                  </span>
                ) : null}
              </button>
            );
          })}
        </div>
      ) : null}
    </div>
  );
}
