'use client';

import { cn } from '@/lib/utils';
import { TrendingDown, TrendingUp, Minus } from 'lucide-react';

interface StepMetrics {
  step: number;
  numbers_preserved: string;
  terms_preserved: string;
  length_ratio: number;
  preservation_score: number;
}

interface Step {
  step: number;
  agent_id: string;
  metrics?: StepMetrics;
  run: number;
}

interface DegradationChartProps {
  steps: Step[];
  currentRun: number;
}

export function DegradationChart({ steps, currentRun }: DegradationChartProps) {
  const runSteps = steps.filter((s) => s.run === currentRun && s.metrics);
  
  if (runSteps.length === 0) {
    return null;
  }

  const scores = runSteps.map((s) => s.metrics?.preservation_score || 0);
  const avgScore = scores.reduce((a, b) => a + b, 0) / scores.length;
  const trend = scores.length > 1 ? scores[scores.length - 1] - scores[0] : 0;

  return (
    <div className="bg-muted/30 border border-border rounded-lg p-4 mb-4">
      <div className="flex items-center justify-between mb-3">
        <h4 className="text-xs font-medium text-muted-foreground">Information Preservation</h4>
        <div className="flex items-center gap-1">
          {trend > 5 ? (
            <TrendingUp className="w-3 h-3 text-emerald-400" />
          ) : trend < -5 ? (
            <TrendingDown className="w-3 h-3 text-rose-400" />
          ) : (
            <Minus className="w-3 h-3 text-zinc-500" />
          )}
          <span className={cn(
            "text-xs font-mono",
            trend > 5 ? "text-emerald-400" : trend < -5 ? "text-rose-400" : "text-zinc-400"
          )}>
            {trend > 0 ? '+' : ''}{trend.toFixed(0)}%
          </span>
        </div>
      </div>
      
      {/* Visual bar chart */}
      <div className="flex items-end gap-1 h-12 mb-2">
        {runSteps.map((step, idx) => {
          const score = step.metrics?.preservation_score || 0;
          const height = Math.max(4, (score / 100) * 48);
          return (
            <div
              key={idx}
              className="flex-1 flex flex-col items-center gap-1"
            >
              <div
                className={cn(
                  "w-full rounded-t transition-all",
                  score >= 80 ? "bg-emerald-500" :
                  score >= 60 ? "bg-amber-500" : "bg-rose-500"
                )}
                style={{ height: `${height}px` }}
              />
            </div>
          );
        })}
      </div>
      
      {/* Labels */}
      <div className="flex gap-1">
        {runSteps.map((step, idx) => (
          <div key={idx} className="flex-1 text-center">
            <span className="text-[9px] text-muted-foreground">
              Agent {step.step + 1}
            </span>
          </div>
        ))}
      </div>
      
      {/* Summary stats */}
      <div className="mt-3 pt-3 border-t border-border flex justify-between text-xs">
        <div>
          <span className="text-muted-foreground">Avg: </span>
          <span className={cn(
            "font-mono",
            avgScore >= 80 ? "text-emerald-400" :
            avgScore >= 60 ? "text-amber-400" : "text-rose-400"
          )}>
            {avgScore.toFixed(1)}%
          </span>
        </div>
        <div>
          <span className="text-muted-foreground">Final: </span>
          <span className={cn(
            "font-mono",
            scores[scores.length - 1] >= 80 ? "text-emerald-400" :
            scores[scores.length - 1] >= 60 ? "text-amber-400" : "text-rose-400"
          )}>
            {scores[scores.length - 1]?.toFixed(1)}%
          </span>
        </div>
      </div>
    </div>
  );
}
