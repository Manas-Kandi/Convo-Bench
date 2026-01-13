'use client';

import { cn, formatDuration } from '@/lib/utils';
import { Bot, User, Clock, ChevronDown, ChevronUp } from 'lucide-react';
import { useState } from 'react';

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
  input: string;
  output: string;
  duration_ms: number;
  run: number;
  metrics?: StepMetrics;
}

interface ConversationViewProps {
  steps: Step[];
  currentRun: number;
}

function MessageBubble({ content, isInput }: { content: string; isInput: boolean }) {
  const [expanded, setExpanded] = useState(false);
  const isLong = content.length > 300;
  const displayContent = isLong && !expanded ? content.slice(0, 300) + '...' : content;

  return (
    <div className={cn(
      'text-sm rounded-lg p-3 whitespace-pre-wrap',
      isInput 
        ? 'text-zinc-300 bg-zinc-800/50 border border-zinc-800'
        : 'bg-accent/5 border border-accent/20'
    )}>
      {displayContent || <span className="text-muted-foreground italic">No response</span>}
      {isLong && (
        <button
          onClick={() => setExpanded(!expanded)}
          className="flex items-center gap-1 text-xs text-accent mt-2 hover:underline"
        >
          {expanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
          {expanded ? 'Show less' : 'Show more'}
        </button>
      )}
    </div>
  );
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
        <div key={`${currentRun}-${step.step}-${idx}`} className="fade-in space-y-3">
          {/* Input message - only show for first step */}
          {idx === 0 && (
            <div className="flex gap-3">
              <div className="w-7 h-7 rounded-full bg-zinc-800 flex items-center justify-center shrink-0">
                <User className="w-3.5 h-3.5 text-zinc-400" />
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-xs text-muted-foreground">Initial Task</span>
                </div>
                <MessageBubble content={step.input} isInput={true} />
              </div>
            </div>
          )}

          {/* Agent response */}
          <div className="flex gap-3">
            <div className="w-7 h-7 rounded-full bg-accent/20 flex items-center justify-center shrink-0">
              <Bot className="w-3.5 h-3.5 text-accent" />
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-1">
                <span className="text-xs font-medium text-accent">
                  {step.agent_id.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                </span>
                <span className="text-[10px] text-muted-foreground flex items-center gap-1">
                  <Clock className="w-2.5 h-2.5" />
                  {formatDuration(step.duration_ms)}
                </span>
                {step.metrics && (
                  <span className={cn(
                    "text-[10px] px-1.5 py-0.5 rounded font-mono",
                    step.metrics.preservation_score >= 80 ? "bg-emerald-500/20 text-emerald-400" :
                    step.metrics.preservation_score >= 60 ? "bg-amber-500/20 text-amber-400" :
                    "bg-rose-500/20 text-rose-400"
                  )}>
                    {step.metrics.preservation_score}% preserved
                  </span>
                )}
              </div>
              <MessageBubble content={step.output} isInput={false} />
              
              {/* Preservation metrics detail */}
              {step.metrics && (
                <div className="mt-2 flex gap-3 text-[10px] text-muted-foreground">
                  <span>Numbers: {step.metrics.numbers_preserved}</span>
                  <span>Terms: {step.metrics.terms_preserved}</span>
                  <span>Length: {step.metrics.length_ratio}x</span>
                </div>
              )}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
