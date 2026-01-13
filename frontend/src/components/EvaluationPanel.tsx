'use client';

import { cn } from '@/lib/utils';

interface Evaluation {
  run: number;
  overall_score: number;
  dimensions: Record<string, number>;
}

interface EvaluationPanelProps {
  evaluations: Evaluation[];
}

const dimensionLabels: Record<string, string> = {
  intent_preservation: 'Intent',
  constraint_adherence: 'Constraints',
  action_correctness: 'Actions',
  coordination_quality: 'Coordination',
  error_propagation: 'Errors',
  information_fidelity: 'Fidelity',
};

function ScoreBar({ score, label }: { score: number; label: string }) {
  const percentage = (score / 5) * 100;
  const color =
    score >= 4 ? 'bg-emerald-500' : score >= 3 ? 'bg-amber-500' : 'bg-rose-500';

  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-mono">{score.toFixed(1)}</span>
      </div>
      <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
        <div
          className={cn('h-full rounded-full transition-all duration-500', color)}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}

export function EvaluationPanel({ evaluations }: EvaluationPanelProps) {
  if (evaluations.length === 0) {
    return (
      <div className="text-center text-muted-foreground text-sm py-8">
        No evaluations yet
      </div>
    );
  }

  // Calculate averages
  const avgOverall =
    evaluations.reduce((sum, e) => sum + e.overall_score, 0) / evaluations.length;

  const dimensionAvgs: Record<string, number> = {};
  const allDimensions = new Set<string>();
  evaluations.forEach((e) => {
    Object.keys(e.dimensions || {}).forEach((d) => allDimensions.add(d));
  });

  allDimensions.forEach((dim) => {
    const scores = evaluations
      .filter((e) => e.dimensions?.[dim] !== undefined)
      .map((e) => e.dimensions[dim]);
    if (scores.length > 0) {
      dimensionAvgs[dim] = scores.reduce((a, b) => a + b, 0) / scores.length;
    }
  });

  return (
    <div className="space-y-6">
      {/* Overall Score */}
      <div className="text-center">
        <div className="text-4xl font-light tracking-tight">
          {avgOverall.toFixed(2)}
        </div>
        <div className="text-xs text-muted-foreground mt-1">
          avg score / 5.0
        </div>
      </div>

      {/* Dimension Scores */}
      <div className="space-y-3">
        {Object.entries(dimensionAvgs).map(([dim, score]) => (
          <ScoreBar
            key={dim}
            score={score}
            label={dimensionLabels[dim] || dim}
          />
        ))}
      </div>

      {/* Run History */}
      <div className="pt-4 border-t border-border">
        <div className="text-xs text-muted-foreground mb-2">Run History</div>
        <div className="flex gap-1 flex-wrap">
          {evaluations.map((e, idx) => {
            const color =
              e.overall_score >= 4
                ? 'bg-emerald-500'
                : e.overall_score >= 3
                ? 'bg-amber-500'
                : 'bg-rose-500';
            return (
              <div
                key={idx}
                className={cn('w-6 h-6 rounded text-[10px] flex items-center justify-center', color)}
                title={`Run ${e.run}: ${e.overall_score.toFixed(1)}`}
              >
                {e.overall_score.toFixed(0)}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
