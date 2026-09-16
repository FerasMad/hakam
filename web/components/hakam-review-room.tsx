"use client";

import { useEffect, useRef, useState } from "react";
import { ErrorState } from "@/components/error-state";
import { Footer } from "@/components/footer";
import { Header } from "@/components/header";
import { HumanReview } from "@/components/human-review";
import { NormalResult } from "@/components/normal-result";
import { ProcessingState } from "@/components/processing-state";
import { UploadStation } from "@/components/upload-station";
import { analysisErrorMessage, uiCopy } from "@/lib/i18n";
import { analysisService, AnalysisServiceError } from "@/lib/services/analysis";
import type { AnalysisResult, Language } from "@/lib/types";
import { validateVideoSelection, type VideoValidationError } from "@/lib/video-validation";

type AppView = "empty" | "selected" | "processing" | "result" | "error";
type ValidationState = VideoValidationError | "preview_error" | "missing" | null;

function validationMessage(language: Language, validation: ValidationState): string | null {
  if (!validation) return null;
  const copy = uiCopy[language];
  if (validation === "one_file_only") return copy.invalidOneFile;
  if (validation === "unsupported_type") return copy.invalidType;
  if (validation === "preview_error") return copy.invalidPreview;
  return copy.invalidMissing;
}

export function HakamReviewRoom() {
  const [language, setLanguage] = useState<Language>("en");
  const [view, setView] = useState<AppView>("empty");
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [previewReady, setPreviewReady] = useState(false);
  const [validation, setValidation] = useState<ValidationState>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [serviceErrorCode, setServiceErrorCode] = useState<string | null>(null);

  const fileInput = useRef<HTMLInputElement>(null);
  const previewUrlRef = useRef<string | null>(null);
  const analysisAbort = useRef<AbortController | null>(null);

  useEffect(() => {
    const savedLanguage = window.localStorage.getItem("hakam-language");
    if (savedLanguage === "ar" || savedLanguage === "en") setLanguage(savedLanguage);

  }, []);

  useEffect(() => {
    document.documentElement.lang = language;
    document.documentElement.dir = language === "ar" ? "rtl" : "ltr";
    window.localStorage.setItem("hakam-language", language);
  }, [language]);

  useEffect(() => {
    return () => {
      analysisAbort.current?.abort();
      if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    };
  }, []);

  const copy = uiCopy[language];
  const currentValidationMessage = validationMessage(language, validation);

  function openPicker() {
    if (fileInput.current) {
      fileInput.current.value = "";
      fileInput.current.click();
    }
  }

  function selectFiles(files: FileList) {
    const checked = validateVideoSelection(files);
    if (!checked.ok) {
      setValidation(checked.error);
      if (file) {
        setResult(null);
        setServiceErrorCode(null);
        setView("selected");
      }
      return;
    }

    analysisAbort.current?.abort();
    analysisAbort.current = null;
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    const nextUrl = URL.createObjectURL(checked.file);
    previewUrlRef.current = nextUrl;
    setFile(checked.file);
    setPreviewUrl(nextUrl);
    setPreviewReady(false);
    setValidation(null);
    setResult(null);
    setServiceErrorCode(null);
    setView("selected");
  }

  function removeVideo() {
    analysisAbort.current?.abort();
    analysisAbort.current = null;
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    previewUrlRef.current = null;
    setFile(null);
    setPreviewUrl(null);
    setPreviewReady(false);
    setValidation(null);
    setResult(null);
    setServiceErrorCode(null);
    setView("empty");
  }

  async function analyze() {
    if (!file || !previewUrl) {
      setValidation("missing");
      return;
    }
    if (!previewReady) return;
    if (analysisAbort.current) return;

    const controller = new AbortController();
    analysisAbort.current = controller;
    setValidation(null);
    setResult(null);
    setServiceErrorCode(null);
    setView("processing");

    try {
      const nextResult = await analysisService.analyze({ file, signal: controller.signal });
      setResult(nextResult);
      setView("result");
      window.requestAnimationFrame(() => {
        window.requestAnimationFrame(() => {
          const headingId = nextResult.status === "abstained" ? "human-review-title" : "vision-result-title";
          document.getElementById(headingId)?.focus();
        });
      });
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") return;
      setServiceErrorCode(error instanceof AnalysisServiceError ? error.code : "ANALYSIS_UNAVAILABLE");
      setView("error");
      window.requestAnimationFrame(() => document.getElementById("analysis-error-title")?.focus());
    } finally {
      if (analysisAbort.current === controller) analysisAbort.current = null;
    }
  }

  const showIntro = view === "empty" || view === "selected";

  return (
    <div className="site-grid min-h-screen bg-carbon">
      <a className="skip-link" href="#main">{copy.skipToContent}</a>
      <Header
        language={language}
        onLanguageChange={() => setLanguage((current) => (current === "en" ? "ar" : "en"))}
        onNewReview={removeVideo}
        showNewReview={Boolean(file)}
      />

      <input
        ref={fileInput}
        type="file"
        accept="video/*"
        className="sr-only"
        aria-label={copy.chooseVideo}
        onChange={(event) => {
          if (event.target.files) selectFiles(event.target.files);
        }}
      />

      <main id="main" tabIndex={-1} className="mx-auto max-w-canvas px-5 sm:px-8 lg:px-12">
        {showIntro ? (
          <section className="max-w-4xl pt-14 sm:pt-20 lg:pt-24" aria-labelledby="page-title">
            <p className="mb-5 text-[0.68rem] font-semibold uppercase tracking-broadcast text-pitch">
              {copy.eyebrow}
            </p>
            <h1 id="page-title" className="text-4xl font-semibold leading-[1.06] tracking-[-0.055em] sm:text-6xl lg:text-7xl">
              {copy.heroTitle}
            </h1>
            <p className="mt-6 max-w-2xl text-base leading-8 text-muted sm:text-lg">{copy.heroBody}</p>
          </section>
        ) : null}

        {view === "empty" || view === "selected" ? (
          <UploadStation
            language={language}
            previewUrl={previewUrl}
            fileName={file?.name ?? null}
            previewReady={previewReady}
            validationMessage={currentValidationMessage}
            onFiles={selectFiles}
            onOpenPicker={openPicker}
            onRemove={removeVideo}
            onAnalyze={analyze}
            onPreviewReady={() => {
              setPreviewReady(true);
              setValidation(null);
            }}
            onPreviewError={() => {
              setPreviewReady(false);
              setValidation("preview_error");
            }}
          />
        ) : null}

        {view === "processing" && previewUrl && file ? (
          <ProcessingState language={language} previewUrl={previewUrl} fileName={file.name} />
        ) : null}

        {view === "result" && result?.status === "completed" && previewUrl && file ? (
          <NormalResult language={language} result={result} previewUrl={previewUrl} fileName={file.name} />
        ) : null}

        {view === "result" && result?.status === "abstained" && previewUrl && file ? (
          <HumanReview
            language={language}
            result={result}
            previewUrl={previewUrl}
            fileName={file.name}
            onReplace={openPicker}
            onReset={removeVideo}
          />
        ) : null}

        {view === "error" && previewUrl && file ? (
          <ErrorState
            language={language}
            previewUrl={previewUrl}
            fileName={file.name}
            message={analysisErrorMessage(language, serviceErrorCode ?? "ANALYSIS_UNAVAILABLE")}
            onRetry={analyze}
            onReplace={openPicker}
          />
        ) : null}
      </main>

      <Footer language={language} />
    </div>
  );
}
