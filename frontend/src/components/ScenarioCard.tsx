'use client';

import { cn, categoryColors } from '@/lib/utils';
import type { Scenario } from '@/lib/api';

interface ScenarioCardProps {
  scenario: Scenario;
  selected: boolean;
  onSelect: () => void;
}

export function ScenarioCard({ scenario, selected, onSelect }: ScenarioCardProps) {
  return (
    <button
      onClick={onSelect}
      className={cn(
        'w-full text-left p-4 rounded-lg border transition-all duration-200',
        'hover:border-zinc-600',
        selected
          ? 'border-accent bg-accent/5'
          : 'border-border bg-muted/50'
      )}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1 min-w-0">
          <h3 className="font-medium text-sm truncate">{scenario.name}</h3>
          <p className="text-xs text-muted-foreground mt-1 line-clamp-2">
            {scenario.description}
          </p>
        </div>
        <span
          className={cn(
            'text-[10px] px-2 py-0.5 rounded-full border shrink-0',
            categoryColors[scenario.category]
          )}
        >
          {scenario.category}
        </span>
      </div>
    </button>
  );
}
