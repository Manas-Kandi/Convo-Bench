'use client';

import { cn } from '@/lib/utils';

interface ConfigPanelProps {
  chainLength: number;
  setChainLength: (v: number) => void;
  complexity: string;
  setComplexity: (v: string) => void;
  numRuns: number;
  setNumRuns: (v: number) => void;
  evaluate: boolean;
  setEvaluate: (v: boolean) => void;
}

export function ConfigPanel({
  chainLength,
  setChainLength,
  complexity,
  setComplexity,
  numRuns,
  setNumRuns,
  evaluate,
  setEvaluate,
}: ConfigPanelProps) {
  return (
    <div className="space-y-4">
      {/* Chain Length */}
      <div>
        <label className="text-xs text-muted-foreground block mb-2">
          Chain Length
        </label>
        <div className="flex gap-2">
          {[2, 3, 4, 5].map((n) => (
            <button
              key={n}
              onClick={() => setChainLength(n)}
              className={cn(
                'flex-1 py-2 text-sm rounded-md border transition-colors',
                chainLength === n
                  ? 'border-accent bg-accent/10 text-accent'
                  : 'border-border hover:border-zinc-600'
              )}
            >
              {n}
            </button>
          ))}
        </div>
      </div>

      {/* Complexity */}
      <div>
        <label className="text-xs text-muted-foreground block mb-2">
          Complexity
        </label>
        <div className="flex gap-2">
          {['simple', 'medium', 'complex'].map((c) => (
            <button
              key={c}
              onClick={() => setComplexity(c)}
              className={cn(
                'flex-1 py-2 text-sm rounded-md border transition-colors capitalize',
                complexity === c
                  ? 'border-accent bg-accent/10 text-accent'
                  : 'border-border hover:border-zinc-600'
              )}
            >
              {c}
            </button>
          ))}
        </div>
      </div>

      {/* Number of Runs */}
      <div>
        <label className="text-xs text-muted-foreground block mb-2">
          Runs
        </label>
        <div className="flex gap-2">
          {[1, 5, 10, 20].map((n) => (
            <button
              key={n}
              onClick={() => setNumRuns(n)}
              className={cn(
                'flex-1 py-2 text-sm rounded-md border transition-colors',
                numRuns === n
                  ? 'border-accent bg-accent/10 text-accent'
                  : 'border-border hover:border-zinc-600'
              )}
            >
              {n}
            </button>
          ))}
        </div>
      </div>

      {/* Evaluate Toggle */}
      <div className="flex items-center justify-between pt-2">
        <span className="text-xs text-muted-foreground">Run Evaluation</span>
        <button
          onClick={() => setEvaluate(!evaluate)}
          className={cn(
            'w-10 h-5 rounded-full transition-colors relative',
            evaluate ? 'bg-accent' : 'bg-zinc-700'
          )}
        >
          <div
            className={cn(
              'absolute top-0.5 w-4 h-4 rounded-full bg-white transition-transform',
              evaluate ? 'translate-x-5' : 'translate-x-0.5'
            )}
          />
        </button>
      </div>
    </div>
  );
}
