import type { SVGProps } from "react";

type IconProps = SVGProps<SVGSVGElement>;

function IconBase({ children, ...props }: IconProps) {
  return (
    <svg
      aria-hidden="true"
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      {...props}
    >
      {children}
    </svg>
  );
}

export function UploadIcon(props: IconProps) {
  return (
    <IconBase {...props}>
      <path d="M12 16V4m0 0 4.5 4.5M12 4 7.5 8.5" />
      <path d="M5 14.5v3A2.5 2.5 0 0 0 7.5 20h9a2.5 2.5 0 0 0 2.5-2.5v-3" />
    </IconBase>
  );
}

export function ReplaceIcon(props: IconProps) {
  return (
    <IconBase {...props}>
      <path d="M20 7h-6a4 4 0 0 0-4 4v1" />
      <path d="m17 4 3 3-3 3M4 17h6a4 4 0 0 0 4-4v-1" />
      <path d="m7 20-3-3 3-3" />
    </IconBase>
  );
}

export function RemoveIcon(props: IconProps) {
  return (
    <IconBase {...props}>
      <path d="M5 7h14M9 7V4h6v3M8 10v7m4-7v7m4-7v7" />
      <path d="m7 7 1 13h8l1-13" />
    </IconBase>
  );
}

export function PlayReviewIcon(props: IconProps) {
  return (
    <IconBase {...props}>
      <path d="M8.5 7.4v9.2L16 12 8.5 7.4Z" />
      <path d="M4 4h3M4 4v3m16-3h-3m3 0v3M4 20h3m-3 0v-3m16 3h-3m3 0v-3" />
    </IconBase>
  );
}

export function ChevronIcon(props: IconProps) {
  return (
    <IconBase {...props}>
      <path d="m8 10 4 4 4-4" />
    </IconBase>
  );
}

export function CopyIcon(props: IconProps) {
  return (
    <IconBase {...props}>
      <rect x="8" y="8" width="11" height="11" rx="2" />
      <path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2" />
    </IconBase>
  );
}

export function CheckIcon(props: IconProps) {
  return (
    <IconBase {...props}>
      <path d="m5 12 4 4L19 6" />
    </IconBase>
  );
}

export function WarningIcon(props: IconProps) {
  return (
    <IconBase {...props}>
      <path d="M12 3 2.8 20h18.4L12 3Z" />
      <path d="M12 9v4m0 3.5v.1" />
    </IconBase>
  );
}

export function ReviewIcon(props: IconProps) {
  return (
    <IconBase {...props}>
      <circle cx="9" cy="8" r="3" />
      <path d="M3.8 19c.5-3.2 2.2-5 5.2-5 1.5 0 2.7.4 3.6 1.2" />
      <path d="m15 17 2 2 4-5" />
    </IconBase>
  );
}

export function DocumentIcon(props: IconProps) {
  return (
    <IconBase {...props}>
      <path d="M7 3h7l4 4v14H7a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2Z" />
      <path d="M14 3v5h4M9 13h6m-6 4h5" />
    </IconBase>
  );
}

export function EyeIcon(props: IconProps) {
  return (
    <IconBase {...props}>
      <path d="M2.5 12s3.4-5 9.5-5 9.5 5 9.5 5-3.4 5-9.5 5-9.5-5-9.5-5Z" />
      <circle cx="12" cy="12" r="2.2" />
    </IconBase>
  );
}

export function ScaleIcon(props: IconProps) {
  return (
    <IconBase {...props}>
      <path d="M12 3v18M6 6h12M4 19h16M7 6 3.5 12h7L7 6Zm10 0-3.5 6h7L17 6Z" />
    </IconBase>
  );
}

export function PenIcon(props: IconProps) {
  return (
    <IconBase {...props}>
      <path d="m4 20 4.3-1 10.8-10.8a2 2 0 0 0-2.8-2.8L5.5 16.2 4 20Z" />
      <path d="m14.8 6.9 2.8 2.8" />
    </IconBase>
  );
}
