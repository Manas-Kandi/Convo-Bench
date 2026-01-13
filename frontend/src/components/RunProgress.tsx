'use client';

import { cn, statusColors } from '@/lib/utils';
import { Loader2, CheckCircle2, XCircle, Circle } from 'lucide-react';

interface RunProgressProps {
  status: string;
  progress: number;
  currentRun: number;
  totalRuns: number;
}

export function RunProgress({ status, progress, currentRun, totalRuns }: RunProgressProps) {
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {status === 'running' && (
            <Loader2 className="w-4 h-4 animate-spin text-accent" />
          )}
          {status === 'completed' && (
            <CheckCircle2 className="w-4 h-4 text-emerald-400" />
          )}
          {status === 'failed' && (
            <XCircle className="w-4 h-4 text-rose-400" />
          )}
          {status === 'pending' && (
            <Circle className="w-4 h-4 text-zinc-500" />
          )}
          <span className={cn('text-sm font-medium', statusColors[status])}>
            {status === 'running' ? `Run ${currentRun}/${totalRuns}` : status}
          </span>
        </div>
        <span className="text-xs text-muted-foreground font-mono">
          {progress}%
        </span>
      </div>
      
      <div className="h-1 bg-zinc-800 rounded-full overflow-hidden">
        <div
          className={cn(
            'h-full rounded-full transition-all duration-300',
            status === 'completed' ? 'bg-emerald-500' :
            status === 'failed' ? 'bg-rose-500' : 'bg-accent'
          )}
          style={{ width: `${progress}%` }}
        />
      </div>
    </div>
  );
}
