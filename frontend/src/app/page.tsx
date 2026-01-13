'use client';

import { useState, useEffect, useCallback } from 'react';
import { Play, History, Settings2, ChevronRight, Zap } from 'lucide-react';
import { cn } from '@/lib/utils';
import { ScenarioCard } from '@/components/ScenarioCard';
import { ModelSelector } from '@/components/ModelSelector';
import { ConversationView } from '@/components/ConversationView';
import { EvaluationPanel } from '@/components/EvaluationPanel';
import { RunProgress } from '@/components/RunProgress';
import { ConfigPanel } from '@/components/ConfigPanel';
import { DegradationChart } from '@/components/DegradationChart';
import {
  fetchScenarios,
  fetchModels,
  createRun,
  createWebSocket,
  type Scenario,
  type Model,
} from '@/lib/api';

type View = 'setup' | 'running';

interface Step {
  step: number;
  agent_id: string;
  input: string;
  output: string;
  duration_ms: number;
  run: number;
}

interface Evaluation {
  run: number;
  overall_score: number;
  dimensions: Record<string, number>;
}

export default function Home() {
  const [view, setView] = useState<View>('setup');
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [models, setModels] = useState<Model[]>([]);
  const [selectedScenario, setSelectedScenario] = useState<string>('');
  const [selectedModel, setSelectedModel] = useState<string>('mock');
  
  // Config
  const [chainLength, setChainLength] = useState(3);
  const [complexity, setComplexity] = useState('medium');
  const [numRuns, setNumRuns] = useState(5);
  const [evaluate, setEvaluate] = useState(true);
  
  // Run state
  const [runId, setRunId] = useState<string | null>(null);
  const [status, setStatus] = useState('pending');
  const [progress, setProgress] = useState(0);
  const [currentRun, setCurrentRun] = useState(1);
  const [steps, setSteps] = useState<Step[]>([]);
  const [evaluations, setEvaluations] = useState<Evaluation[]>([]);

  // Load initial data
  useEffect(() => {
    fetchScenarios().then((s) => {
      setScenarios(s);
      if (s.length > 0) setSelectedScenario(s[0].id);
    });
    fetchModels().then((data) => {
      setModels(data.models);
    });
  }, []);

  // WebSocket connection with polling fallback
  useEffect(() => {
    if (!runId) return;

    let ws: WebSocket | null = null;
    let lastUpdateIndex = 0;

    // Track which steps we've already processed to avoid duplicates
    const processedSteps = new Set<string>();
    const processedEvals = new Set<number>();

    // Process updates from either WebSocket or polling
    const processUpdate = (data: any) => {
      if (data.type === 'progress') {
        setProgress(data.progress);
        setCurrentRun(data.run);
      } else if (data.type === 'step') {
        // Create unique key to prevent duplicates
        const stepKey = `${data.run}-${data.step}-${data.agent_id}`;
        if (!processedSteps.has(stepKey)) {
          processedSteps.add(stepKey);
          setSteps((prev) => [...prev, data]);
        }
      } else if (data.type === 'evaluation') {
        if (!processedEvals.has(data.run)) {
          processedEvals.add(data.run);
          setEvaluations((prev) => [...prev, data]);
        }
      } else if (data.type === 'complete') {
        setStatus('completed');
        setProgress(100);
      } else if (data.type === 'error') {
        setStatus('failed');
      }
    };

    // Try WebSocket
    try {
      ws = createWebSocket(runId);
      
      ws.onmessage = (event) => {
        if (event.data === 'pong') return;
        const data = JSON.parse(event.data);
        processUpdate(data);
      };

      ws.onerror = () => {
        console.log('WebSocket error, using polling fallback');
      };
    } catch (e) {
      console.log('WebSocket failed, using polling');
    }

    // Polling fallback - always poll to ensure we get updates
    const pollInterval = setInterval(async () => {
      try {
        const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/runs/${runId}`);
        const data = await res.json();
        
        // Process any new updates
        if (data.updates && data.updates.length > lastUpdateIndex) {
          const newUpdates = data.updates.slice(lastUpdateIndex);
          newUpdates.forEach(processUpdate);
          lastUpdateIndex = data.updates.length;
        }
        
        // Update status
        if (data.status === 'completed' || data.status === 'failed') {
          setStatus(data.status);
          if (data.status === 'completed') setProgress(100);
          clearInterval(pollInterval);
        }
      } catch (e) {
        // Ignore polling errors
      }
    }, 1000);

    return () => {
      clearInterval(pollInterval);
      if (ws) ws.close();
    };
  }, [runId]);

  const handleStart = useCallback(async () => {
    // Reset state first
    setSteps([]);
    setEvaluations([]);
    setProgress(0);
    setCurrentRun(1);
    setStatus('running');
    setView('running');

    try {
      const result = await createRun({
        scenario: {
          scenario_type: selectedScenario,
          chain_length: chainLength,
          complexity,
          num_constraints: 3,
          num_agents: chainLength,
        },
        model: selectedModel,
        num_runs: numRuns,
        evaluate,
      });

      setRunId(result.run_id);
    } catch (e) {
      console.error('Failed to start run:', e);
      setStatus('failed');
    }
  }, [selectedScenario, selectedModel, chainLength, complexity, numRuns, evaluate]);

  const handleReset = () => {
    setView('setup');
    setRunId(null);
    setStatus('pending');
    setProgress(0);
    setSteps([]);
    setEvaluations([]);
  };

  const selectedScenarioData = scenarios.find((s) => s.id === selectedScenario);

  return (
    <main className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border">
        <div className="max-w-7xl mx-auto px-6 h-14 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Zap className="w-5 h-5 text-accent" />
            <span className="font-semibold tracking-tight">ConvoBench</span>
          </div>
          {view === 'running' && (
            <button
              onClick={handleReset}
              className="text-sm text-muted-foreground hover:text-foreground transition-colors"
            >
              New Test
            </button>
          )}
        </div>
      </header>

      {view === 'setup' ? (
        <div className="max-w-7xl mx-auto px-6 py-8">
          <div className="grid grid-cols-12 gap-8">
            {/* Scenarios */}
            <div className="col-span-5">
              <h2 className="text-sm font-medium mb-4 flex items-center gap-2">
                <History className="w-4 h-4 text-muted-foreground" />
                Scenario
              </h2>
              <div className="space-y-2 max-h-[60vh] overflow-y-auto pr-2">
                {scenarios.map((scenario) => (
                  <ScenarioCard
                    key={scenario.id}
                    scenario={scenario}
                    selected={selectedScenario === scenario.id}
                    onSelect={() => setSelectedScenario(scenario.id)}
                  />
                ))}
              </div>
            </div>

            {/* Config */}
            <div className="col-span-3">
              <h2 className="text-sm font-medium mb-4 flex items-center gap-2">
                <Settings2 className="w-4 h-4 text-muted-foreground" />
                Configuration
              </h2>
              <div className="bg-muted/50 border border-border rounded-lg p-4">
                <ConfigPanel
                  chainLength={chainLength}
                  setChainLength={setChainLength}
                  complexity={complexity}
                  setComplexity={setComplexity}
                  numRuns={numRuns}
                  setNumRuns={setNumRuns}
                  evaluate={evaluate}
                  setEvaluate={setEvaluate}
                />
              </div>
            </div>

            {/* Models */}
            <div className="col-span-4">
              <h2 className="text-sm font-medium mb-4">Model</h2>
              <ModelSelector
                models={models}
                selected={selectedModel}
                onSelect={setSelectedModel}
              />
            </div>
          </div>

          {/* Start Button */}
          <div className="mt-8 flex justify-center">
            <button
              onClick={handleStart}
              disabled={!selectedScenario}
              className={cn(
                'flex items-center gap-2 px-8 py-3 rounded-lg font-medium transition-all',
                'bg-accent hover:bg-accent/90 text-white',
                'disabled:opacity-50 disabled:cursor-not-allowed'
              )}
            >
              <Play className="w-4 h-4" />
              Start Benchmark
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      ) : (
        <div className="max-w-7xl mx-auto px-6 py-6">
          {/* Run Info */}
          <div className="mb-6">
            <div className="flex items-center gap-3 mb-4">
              <h1 className="text-lg font-medium">
                {selectedScenarioData?.name}
              </h1>
              <span className="text-xs text-muted-foreground px-2 py-0.5 bg-muted rounded">
                {selectedModel}
              </span>
            </div>
            <RunProgress
              status={status}
              progress={progress}
              currentRun={currentRun}
              totalRuns={numRuns}
            />
          </div>

          {/* Main Content */}
          <div className="grid grid-cols-12 gap-6">
            {/* Conversation */}
            <div className="col-span-8">
              <div className="bg-muted/30 border border-border rounded-lg h-[65vh] flex flex-col">
                <div className="px-4 py-3 border-b border-border flex items-center justify-between">
                  <span className="text-sm font-medium">Conversation</span>
                  {steps.length > 0 && (
                    <div className="flex gap-1">
                      {Array.from({ length: numRuns }, (_, i) => i + 1).map((r) => (
                        <button
                          key={r}
                          onClick={() => setCurrentRun(r)}
                          className={cn(
                            'w-6 h-6 text-xs rounded transition-colors',
                            currentRun === r
                              ? 'bg-accent text-white'
                              : 'bg-zinc-800 text-zinc-400 hover:bg-zinc-700'
                          )}
                        >
                          {r}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
                <ConversationView steps={steps} currentRun={currentRun} />
              </div>
            </div>

            {/* Evaluation & Metrics */}
            <div className="col-span-4 space-y-4">
              {/* Degradation Chart */}
              <DegradationChart steps={steps} currentRun={currentRun} />
              
              {/* Evaluation */}
              <div className="bg-muted/30 border border-border rounded-lg p-4">
                <h3 className="text-sm font-medium mb-4">Evaluation</h3>
                <EvaluationPanel evaluations={evaluations} />
              </div>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}
