import { useState } from "react";
import {
  FiAlertTriangle,
  FiCheckCircle,
  FiInfo,
  FiLink,
  FiLoader,
  FiRefreshCw,
  FiSearch,
  FiShield,
  FiXCircle,
  FiZap,
} from "react-icons/fi";

import { analyzeUrl } from "../services/api";

function UrlScanner() {
  const [url, setUrl] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleAnalyze = async (event) => {
    event.preventDefault();

    const cleanUrl = url.trim();

    if (!cleanUrl) {
      setError("Please enter a URL to analyze.");
      return;
    }

    try {
      setLoading(true);
      setError("");
      setResult(null);

      const data = await analyzeUrl(cleanUrl);

      setResult(data);
    } catch (err) {
      console.error("URL analysis error:", err);

      const message =
        err?.response?.data?.detail ||
        "Unable to analyze this URL. Make sure the DeepShield backend is running.";

      setError(message);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setUrl("");
    setResult(null);
    setError("");
  };

  const isThreat = result?.status === "THREAT";

  const getRiskStyle = (level) => {
    switch (level) {
      case "SAFE":
        return "border-emerald-500/30 bg-emerald-500/10 text-emerald-400";

      case "LOW":
        return "border-cyan-500/30 bg-cyan-500/10 text-cyan-400";

      case "MEDIUM":
        return "border-yellow-500/30 bg-yellow-500/10 text-yellow-300";

      case "HIGH":
        return "border-orange-500/30 bg-orange-500/10 text-orange-400";

      case "CRITICAL":
        return "border-red-500/30 bg-red-500/10 text-red-400";

      default:
        return "border-slate-700 bg-slate-800 text-slate-300";
    }
  };

  return (
    <div className="min-h-full text-slate-100">
      <div className="mx-auto max-w-7xl">

        {/* PAGE HEADER */}

        <div className="mb-8">
          <div className="flex items-center gap-4">
            <div className="flex h-12 w-12 items-center justify-center rounded-xl border border-cyan-500/20 bg-cyan-500/10">
              <FiShield className="text-2xl text-cyan-400" />
            </div>

            <div>
              <h1 className="text-2xl font-semibold text-white">
                URL Security Scanner
              </h1>

              <p className="mt-1 text-sm text-slate-400">
                Analyze suspicious URLs using the DeepShield AI two-stage
                detection pipeline.
              </p>
            </div>
          </div>
        </div>

        {/* URL INPUT */}

        <section className="rounded-2xl border border-slate-800 bg-[#0d131c] p-6">
          <div className="mb-5 flex items-center gap-2 text-sm font-semibold text-slate-200">
            <FiSearch className="text-cyan-400" />
            Analyze URL
          </div>

          <form
            onSubmit={handleAnalyze}
            className="flex flex-col gap-3 lg:flex-row"
          >
            <div className="relative flex-1">
              <FiLink className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-500" />

              <input
                type="text"
                value={url}
                onChange={(event) => {
                  setUrl(event.target.value);
                  setError("");
                }}
                placeholder="https://example.com"
                className="h-12 w-full rounded-xl border border-slate-700 bg-[#080d14] pl-11 pr-4 text-sm text-white outline-none transition placeholder:text-slate-600 focus:border-cyan-500"
              />
            </div>

            <button
              type="submit"
              disabled={loading}
              className="flex h-12 items-center justify-center gap-2 rounded-xl bg-cyan-500 px-6 text-sm font-semibold text-slate-950 transition hover:bg-cyan-400 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {loading ? (
                <>
                  <FiLoader className="animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  <FiZap />
                  Analyze URL
                </>
              )}
            </button>

            {(result || error) && (
              <button
                type="button"
                onClick={handleReset}
                className="flex h-12 items-center justify-center gap-2 rounded-xl border border-slate-700 px-5 text-sm text-slate-300 transition hover:bg-slate-800"
              >
                <FiRefreshCw />
                Reset
              </button>
            )}
          </form>

          <div className="mt-4 flex items-start gap-2 text-xs leading-5 text-slate-500">
            <FiInfo className="mt-0.5 shrink-0" />

            <span>
              DeepShield evaluates lexical and structural URL characteristics
              using trained machine-learning models.
            </span>
          </div>

          {error && (
            <div className="mt-5 flex items-start gap-3 rounded-xl border border-red-500/20 bg-red-500/10 p-4">
              <FiXCircle className="mt-0.5 shrink-0 text-red-400" />

              <div>
                <p className="text-sm font-semibold text-red-300">
                  Analysis failed
                </p>

                <p className="mt-1 text-sm text-red-300/80">
                  {error}
                </p>
              </div>
            </div>
          )}
        </section>

        {/* RESULTS */}

        {result && (
          <div className="mt-6 space-y-6">

            {/* MAIN RESULT */}

            <section
              className={`rounded-2xl border p-6 ${
                isThreat
                  ? "border-red-500/20 bg-red-500/[0.04]"
                  : "border-emerald-500/20 bg-emerald-500/[0.04]"
              }`}
            >
              <div className="flex flex-col gap-6 lg:flex-row lg:items-center lg:justify-between">
                <div className="flex items-start gap-4">
                  <div
                    className={`flex h-12 w-12 shrink-0 items-center justify-center rounded-xl ${
                      isThreat
                        ? "bg-red-500/10 text-red-400"
                        : "bg-emerald-500/10 text-emerald-400"
                    }`}
                  >
                    {isThreat ? (
                      <FiAlertTriangle className="text-2xl" />
                    ) : (
                      <FiCheckCircle className="text-2xl" />
                    )}
                  </div>

                  <div>
                    <p className="text-xs uppercase tracking-[0.18em] text-slate-500">
                      AI Analysis Result
                    </p>

                    <h2
                      className={`mt-1 text-2xl font-semibold ${
                        isThreat
                          ? "text-red-400"
                          : "text-emerald-400"
                      }`}
                    >
                      {result.status}
                    </h2>

                    <p className="mt-2 break-all text-sm text-slate-400">
                      {result.url}
                    </p>
                  </div>
                </div>

                <div
                  className={`w-fit rounded-full border px-4 py-2 text-sm font-semibold ${getRiskStyle(
                    result.risk_level
                  )}`}
                >
                  {result.risk_level} RISK
                </div>
              </div>
            </section>

            {/* METRICS */}

            <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
              <MetricCard
                title="Threat Score"
                value={`${result.threat_score}/100`}
                description={`Decision threshold: ${result.decision_threshold}/100`}
              />

              <MetricCard
                title="Risk Score"
                value={`${result.risk_score}/100`}
                description="DeepShield security risk score"
              />

              <MetricCard
                title="Threat Type"
                value={
                  result.threat_type
                    ? result.threat_type.toUpperCase()
                    : "NONE"
                }
                description={
                  result.threat_type_confidence
                    ? `${result.threat_type_confidence}% confidence`
                    : "No threat type required"
                }
              />

              <MetricCard
                title="Detector"
                value="V3"
                description="Hierarchical AI detection"
              />
            </div>

            {/* THREAT TYPE SCORES */}

            {result.status === "THREAT" &&
              Object.keys(result.threat_type_scores || {}).length > 0 && (
                <Panel
                  title="Threat Classification"
                  icon={<FiShield />}
                >
                  <div className="space-y-5">
                    {Object.entries(result.threat_type_scores).map(
                      ([label, score]) => (
                        <ThreatBar
                          key={label}
                          label={label}
                          score={score}
                        />
                      )
                    )}
                  </div>
                </Panel>
              )}

            {/* SECURITY INFORMATION */}

            <div className="grid gap-6 xl:grid-cols-2">

              <Panel
                title="Security Signals"
                icon={<FiAlertTriangle />}
              >
                <div className="space-y-3">
                  {(result.security_signals || []).map(
                    (signal, index) => (
                      <div
                        key={`${signal}-${index}`}
                        className="flex gap-3 rounded-xl border border-slate-800 bg-[#080d14] p-4"
                      >
                        <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-cyan-500/10 text-xs font-semibold text-cyan-400">
                          {index + 1}
                        </div>

                        <p className="text-sm leading-6 text-slate-300">
                          {signal}
                        </p>
                      </div>
                    )
                  )}
                </div>
              </Panel>

              <Panel
                title="Recommended Action"
                icon={<FiZap />}
              >
                <div
                  className={`rounded-xl border p-5 ${
                    isThreat
                      ? "border-red-500/20 bg-red-500/[0.05]"
                      : "border-emerald-500/20 bg-emerald-500/[0.05]"
                  }`}
                >
                  <p className="text-sm leading-7 text-slate-300">
                    {result.recommended_action}
                  </p>
                </div>

                <div className="mt-5 rounded-xl border border-slate-800 bg-[#080d14] p-5">
                  <p className="mb-4 text-xs font-semibold uppercase tracking-[0.15em] text-slate-500">
                    AI Detection Pipeline
                  </p>

                  <PipelineRow
                    stage="Stage 1"
                    model={result.models?.stage_1 || "Unknown"}
                    description="Benign vs Threat detection"
                  />

                  <div className="my-4 h-px bg-slate-800" />

                  <PipelineRow
                    stage="Stage 2"
                    model={result.models?.stage_2 || "Unknown"}
                    description="Threat type classification"
                  />
                </div>
              </Panel>

            </div>

            {/* DISCLAIMER */}

            <div className="flex gap-3 rounded-xl border border-slate-800 bg-[#0d131c] p-4 text-xs leading-5 text-slate-500">
              <FiInfo className="mt-0.5 shrink-0" />

              <p>
                Threat Score is a machine-learning security score and should not
                be interpreted as a guaranteed real-world probability.
                DeepShield AI should be used as one layer of security analysis.
              </p>
            </div>

          </div>
        )}

      </div>
    </div>
  );
}

function MetricCard({ title, value, description }) {
  return (
    <div className="rounded-2xl border border-slate-800 bg-[#0d131c] p-5">
      <p className="text-xs font-medium uppercase tracking-[0.14em] text-slate-500">
        {title}
      </p>

      <p className="mt-3 text-xl font-semibold text-white">
        {value}
      </p>

      <p className="mt-2 text-xs text-slate-500">
        {description}
      </p>
    </div>
  );
}

function Panel({ title, icon, children }) {
  return (
    <section className="rounded-2xl border border-slate-800 bg-[#0d131c] p-5 lg:p-6">
      <div className="mb-5 flex items-center gap-2 text-sm font-semibold text-slate-200">
        <span className="text-cyan-400">
          {icon}
        </span>

        {title}
      </div>

      {children}
    </section>
  );
}

function ThreatBar({ label, score }) {
  const numericScore = Number(score) || 0;

  const safeScore = Math.max(
    0,
    Math.min(100, numericScore)
  );

  return (
    <div>
      <div className="mb-2 flex items-center justify-between">
        <span className="text-sm font-medium capitalize text-slate-300">
          {label}
        </span>

        <span className="text-sm font-semibold text-white">
          {safeScore.toFixed(2)}%
        </span>
      </div>

      <div className="h-2 overflow-hidden rounded-full bg-slate-800">
        <div
          className="h-full rounded-full bg-cyan-400 transition-all duration-500"
          style={{
            width: `${safeScore}%`,
          }}
        />
      </div>
    </div>
  );
}

function PipelineRow({ stage, model, description }) {
  return (
    <div className="flex items-start justify-between gap-6">
      <div>
        <p className="text-sm font-medium text-slate-200">
          {stage}
        </p>

        <p className="mt-1 text-xs text-slate-500">
          {description}
        </p>
      </div>

      <p className="text-right text-xs font-medium text-cyan-400">
        {model}
      </p>
    </div>
  );
}

export default UrlScanner;