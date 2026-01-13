'use client';

import { cn } from '@/lib/utils';
import type { Model } from '@/lib/api';
import { Cpu, Sparkles } from 'lucide-react';

interface ModelSelectorProps {
  models: Model[];
  selected: string;
  onSelect: (id: string) => void;
}

export function ModelSelector({ models, selected, onSelect }: ModelSelectorProps) {
  return (
    <div className="space-y-2">
      {models.map((model) => (
        <button
          key={model.id}
          onClick={() => model.available && onSelect(model.id)}
          disabled={!model.available}
          className={cn(
            'w-full flex items-center gap-3 p-3 rounded-lg border transition-all',
            model.available ? 'hover:border-zinc-600' : 'opacity-40 cursor-not-allowed',
            selected === model.id
              ? 'border-accent bg-accent/5'
              : 'border-border bg-muted/50'
          )}
        >
          <div className="w-8 h-8 rounded-md bg-zinc-800 flex items-center justify-center">
            <Cpu className="w-4 h-4 text-zinc-400" />
          </div>
          <div className="flex-1 text-left">
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium">{model.name}</span>
              {model.supports_reasoning && (
                <Sparkles className="w-3 h-3 text-amber-400" />
              )}
            </div>
            <span className="text-xs text-muted-foreground">{model.provider}</span>
          </div>
          {!model.available && (
            <span className="text-[10px] text-zinc-500">No API key</span>
          )}
        </button>
      ))}
    </div>
  );
}
