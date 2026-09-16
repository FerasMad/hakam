export type VideoValidationError = "one_file_only" | "unsupported_type";

export type VideoValidationResult =
  | { ok: true; file: File }
  | { ok: false; error: VideoValidationError };

export function validateVideoSelection(files: FileList | File[]): VideoValidationResult {
  const selected = Array.from(files);
  if (selected.length !== 1) {
    return { ok: false, error: "one_file_only" };
  }

  const [file] = selected;
  if (!file.type.startsWith("video/")) {
    return { ok: false, error: "unsupported_type" };
  }

  const probe = document.createElement("video");
  if (!probe.canPlayType(file.type)) {
    return { ok: false, error: "unsupported_type" };
  }

  return { ok: true, file };
}
