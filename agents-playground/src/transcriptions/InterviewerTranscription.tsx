import React from "react";
import { useAgent } from "../hooks/use-agent";
import { cn } from "../lib/util";
import { Track } from "livekit-client";

interface InterviewerTranscriptionProps {
  className?: string;
}

export function InterviewerTranscription({ className }: InterviewerTranscriptionProps) {
  const { displayTranscriptions } = useAgent();

  return (
    <div className={cn("flex flex-col gap-2", className)}>
      {displayTranscriptions
        .filter((transcription) => !transcription.participant?.isAgent && transcription?.publication?.source === Track.Source.ScreenShareAudio)
        .map((transcription) => (
          <div
            key={transcription.segment.id}
            className={cn("rounded-lg bg-muted p-3 text-sm")}
          >
            <div className={cn("font-medium text-foreground")}>
              {transcription.participant?.identity || "Unknown"}
            </div>
            <div className={cn("text-muted-foreground")}>
              {transcription.segment.text}
            </div>
          </div>
        ))}
    </div>
  );
}