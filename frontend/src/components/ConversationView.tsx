'use client';

import { cn, formatDuration } from '@/lib/utils';
import { Bot, User, Clock } from 'lucide-react';

interface Step {
  step: number;
  agent_id: string;
  input: string;
  output: string;
  duration_ms: number;
  run: number;
}

interface ConversationViewProps {
  steps: Step[];
  currentRun: number;
}

export function ConversationView({ steps, currentRun }: ConversationViewProps) {
  const runSteps = steps.filter((s) => s.run === currentRun);

  if (runSteps.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center text-muted-foreground text-sm">
        Waiting for conversation...
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto space-y-4 p-4">
      {runSteps.map((step, idx) => (
        <div key={idx} className="fade-in space-y-3">
          {/* Input message */}
          <div className="flex gap-3">
            <div className="w-7 h-7 rounded-full bg-zinc-800 flex items-center justify-center shrink-0">
              {idx === 0 ? (
                <User className="w-3.5 h-3.5 text-zinc-400" />
              ) : (
                <Bot className="w-3.5 h-3.5 text-zinc-400" />
              )}
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-1">
                <span className="text-xs text-muted-foreground">
                  {idx === 0 ? 'Initial' : `Agent ${step.step}`}
                </span>
              </div>
              <div className="text-sm text-zinc-300 bg-zinc-800/50 rounded-lg p-3 border border-zinc-800">
                {step.input || '...'}
              </div>
            </div>
          </div>

          {/* Output message */}
          <div className="flex gap-3 pl-10">
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-1">
                <span className="text-xs font-medium text-accent">
                  {step.agent_id}
                </span>
                <span className="text-[10px] text-muted-foreground flex items-center gap-1">
                  <Clock className="w-2.5 h-2.5" />
                  {formatDuration(step.duration_ms)}
                </span>
              </div>
              <div className="text-sm bg-accent/5 border border-accent/20 rounded-lg p-3">
                {step.output || '...'}
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
