export const MAX_VIDEO_BYTES = 200 * 1024 * 1024;

export type VideoValidationError = "one_file_only" | "unsupported_type" | "too_large";

export type VideoValidationResult =
  | { ok: true; file: File }
  | { ok: false; error: VideoValidationError };

export function validateVideoSelection(files: FileList | File[]): VideoValidationResult {
  const selected = Array.from(files);
  if (selected.length !== 1) {
    return { ok: false, error: "one_file_only" };
  }

  const [file] = selected;
  if (file.size > MAX_VIDEO_BYTES) {
    return { ok: false, error: "too_large" };
  }
  if (!file.type.startsWith("video/")) {
    return { ok: false, error: "unsupported_type" };
  }

  const probe = document.createElement("video");
  if (!probe.canPlayType(file.type)) {
    return { ok: false, error: "unsupported_type" };
  }

  return { ok: true, file };
}
