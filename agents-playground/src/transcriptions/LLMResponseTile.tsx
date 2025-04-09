import React from "react";
import { useLLMStream } from "../hooks/use-llm-stream";
import { cn } from "../lib/util";

interface LLMResponseTileProps {
  className?: string;
}

export function LLMResponseTile({ className }: LLMResponseTileProps) {
  const { llmResponses } = useLLMStream();

  return (
    <div className={cn("flex flex-col gap-2", className)}>
      {llmResponses.length === 0 ? (
        <div className="text-sm text-muted-foreground">
          Waiting for assistant response...
        </div>
      ) : (
        llmResponses.map((response, index) => (
          <div
            key={`${response.timestamp}-${index}`}
            className="rounded-lg bg-primary/10 p-3 text-sm"
          >
            <div className="font-medium text-foreground mb-1">
              Assistant
            </div>
            <div className="text-primary whitespace-pre-wrap">
              {response.text}
            </div>
          </div>
        ))
      )}
    </div>
  );
}
