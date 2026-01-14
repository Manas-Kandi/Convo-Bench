'use client';

import { useState, useEffect } from 'react';
import { Play, RefreshCw, CheckCircle, XCircle, Clock, Zap, ArrowLeft } from 'lucide-react';
import Link from 'next/link';
import {
  triggerSweep,
  fetchSweeps,
  fetchSweep,
  type SweepSummary,
  type SweepResult,
} from '@/lib/api';

export default function BenchmarksPage() {
  const [sweeps, setSweeps] = useState<SweepSummary[]>([]);
  const [selectedSweep, setSelectedSweep] = useState<SweepResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [sweepRunning, setSweepRunning] = useState(false);
  const [runningSweepId, setRunningSweepId] = useState<string | null>(null);

  const loadSweeps = async () => {
    const data = await fetchSweeps();
    setSweeps(data);
  };

  useEffect(() => {
    loadSweeps();
  }, []);

  // Poll for running sweep completion
  useEffect(() => {
    if (!runningSweepId) return;

    const interval = setInterval(async () => {
      try {
        const result = await fetchSweep(runningSweepId);
        if (result.finished_at) {
          setSweepRunning(false);
          setRunningSweepId(null);
          setSelectedSweep(result);
          loadSweeps();
        }
      } catch {
        // Still running or not found yet
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [runningSweepId]);

  const handleTriggerSweep = async () => {
    setLoading(true);
    setSweepRunning(true);
    try {
      const result = await triggerSweep();
      setRunningSweepId(result.sweep_id);
    } catch (e) {
      console.error('Failed to trigger sweep:', e);
      setSweepRunning(false);
    }
    setLoading(false);
  };

  const handleSelectSweep = async (sweepId: string) => {
    setLoading(true);
    try {
      const result = await fetchSweep(sweepId);
      setSelectedSweep(result);
    } catch (e) {
      console.error('Failed to fetch sweep:', e);
    }
    setLoading(false);
  };

  return (
    <main className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border">
        <div className="max-w-7xl mx-auto px-6 h-14 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link href="/" className="flex items-center gap-2 text-muted-foreground hover:text-foreground transition-colors">
              <ArrowLeft className="w-4 h-4" />
              Back
            </Link>
            <div className="flex items-center gap-2">
              <Zap className="w-5 h-5 text-accent" />
              <span className="font-semibold tracking-tight">ConvoBench</span>
              <span className="text-muted-foreground">/ Benchmarks</span>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Trigger Button */}
        <div className="mb-8 flex items-center gap-4">
          <button
            onClick={handleTriggerSweep}
            disabled={loading || sweepRunning}
            className="flex items-center gap-2 px-6 py-3 rounded-lg font-medium bg-accent hover:bg-accent/90 text-white disabled:opacity-50 disabled:cursor-not-allowed transition-all"
          >
            {sweepRunning ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin" />
                Running Comprehensive Sweep...
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                Run Comprehensive Benchmark
              </>
            )}
          </button>
          <button
            onClick={loadSweeps}
            className="flex items-center gap-2 px-4 py-2 rounded-lg text-muted-foreground hover:text-foreground border border-border hover:border-foreground/20 transition-all"
          >
            <RefreshCw className="w-4 h-4" />
            Refresh
          </button>
        </div>

        <div className="grid grid-cols-12 gap-8">
          {/* Sweep List */}
          <div className="col-span-4">
            <h2 className="text-sm font-medium mb-4">Previous Sweeps</h2>
            <div className="space-y-2 max-h-[70vh] overflow-y-auto">
              {sweeps.length === 0 ? (
                <p className="text-muted-foreground text-sm">No sweeps yet. Run one to get started.</p>
              ) : (
                sweeps.map((s) => (
                  <button
                    key={s.sweep_id}
                    onClick={() => handleSelectSweep(s.sweep_id)}
                    className={`w-full text-left px-4 py-3 rounded-lg border transition-all ${
                      selectedSweep?.sweep_id === s.sweep_id
                        ? 'border-accent bg-accent/10'
                        : 'border-border hover:border-accent/50'
                    }`}
                  >
                    <div className="font-mono text-sm">{s.sweep_id}</div>
                  </button>
                ))
              )}
            </div>
          </div>

          {/* Sweep Results */}
          <div className="col-span-8">
            {selectedSweep ? (
              <div>
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-sm font-medium">Results: {selectedSweep.sweep_id}</h2>
                  <div className="text-sm text-muted-foreground">
                    {selectedSweep.completed}/{selectedSweep.total_combos} completed
                  </div>
                </div>

                <div className="bg-muted/30 border border-border rounded-lg overflow-hidden">
                  <table className="w-full text-sm">
                    <thead className="bg-muted/50">
                      <tr>
                        <th className="text-left px-4 py-2 font-medium">Scenario</th>
                        <th className="text-left px-4 py-2 font-medium">Baseline</th>
                        <th className="text-left px-4 py-2 font-medium">Model</th>
                        <th className="text-left px-4 py-2 font-medium">Status</th>
                        <th className="text-left px-4 py-2 font-medium">Score</th>
                      </tr>
                    </thead>
                    <tbody>
                      {selectedSweep.results.map((r, i) => (
                        <tr key={i} className="border-t border-border">
                          <td className="px-4 py-2">{r.scenario}</td>
                          <td className="px-4 py-2">{r.baseline}</td>
                          <td className="px-4 py-2">{r.model}</td>
                          <td className="px-4 py-2">
                            {r.status === 'completed' ? (
                              <span className="flex items-center gap-1 text-green-500">
                                <CheckCircle className="w-3 h-3" /> Completed
                              </span>
                            ) : (
                              <span className="flex items-center gap-1 text-red-500">
                                <XCircle className="w-3 h-3" /> Failed
                              </span>
                            )}
                          </td>
                          <td className="px-4 py-2">
                            {r.aggregate_metrics?.scores?.overall?.mean?.toFixed(2) ?? '-'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <div className="mt-4 text-xs text-muted-foreground">
                  Started: {selectedSweep.started_at} | Finished: {selectedSweep.finished_at ?? 'In progress'}
                </div>
              </div>
            ) : (
              <div className="flex items-center justify-center h-64 text-muted-foreground">
                Select a sweep to view results, or run a new comprehensive benchmark.
              </div>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}
