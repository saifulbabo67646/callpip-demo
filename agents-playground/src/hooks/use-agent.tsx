import React, { createContext, useContext, useState, useEffect } from "react";
import {
  useMaybeRoomContext,
  useVoiceAssistant,
  useLocalParticipant,
} from "@livekit/components-react";
import {
  RoomEvent,
  TranscriptionSegment,
  Participant,
  TrackPublication,
  RemoteParticipant,
  type RpcInvocationData,
} from "livekit-client";

interface Transcription {
  segment: TranscriptionSegment;
  participant?: Participant;
  publication?: TrackPublication;
}

interface AgentContextType {
  displayTranscriptions: Transcription[];
  agent?: RemoteParticipant;
}

const AgentContext = createContext<AgentContextType | undefined>(undefined);

export function AgentProvider({ children }: { children: React.ReactNode }) {
  const room = useMaybeRoomContext();
  const { agent } = useVoiceAssistant();
  const { localParticipant } = useLocalParticipant();
  const [rawSegments, setRawSegments] = useState<{
    [id: string]: Transcription;
  }>({});
  const [displayTranscriptions, setDisplayTranscriptions] = useState<
    Transcription[]
  >([]);

  useEffect(() => {
    if (!room) {
      return;
    }
    const updateRawSegments = (
      segments: TranscriptionSegment[],
      participant?: Participant,
      publication?: TrackPublication,
    ) => {
      setRawSegments((prev) => {
        const newSegments = { ...prev };
        for (const segment of segments) {
          newSegments[segment.id] = { segment, participant, publication };
        }
        return newSegments;
      });
    };
    room.on(RoomEvent.TranscriptionReceived, updateRawSegments);

    return () => {
      room.off(RoomEvent.TranscriptionReceived, updateRawSegments);
    };
  }, [room]);

  useEffect(() => {
    const sorted = Object.values(rawSegments).sort(
      (a, b) =>
        (a.segment.firstReceivedTime ?? 0) - (b.segment.firstReceivedTime ?? 0),
    );
    const mergedSorted = sorted.reduce((acc, current) => {
      if (acc.length === 0) {
        return [current];
      }

      const last = acc[acc.length - 1];
      if (
        last.participant === current.participant &&
        last.participant?.isAgent &&
        (current.segment.firstReceivedTime ?? 0) -
          (last.segment.lastReceivedTime ?? 0) <=
          1000 &&
        !last.segment.id.startsWith("status-") &&
        !current.segment.id.startsWith("status-")
      ) {
        return [
          ...acc.slice(0, -1),
          {
            ...current,
            segment: {
              ...current.segment,
              text: `${last.segment.text} ${current.segment.text}`,
              id: current.segment.id,
              firstReceivedTime: last.segment.firstReceivedTime,
            },
          },
        ];
      } else {
        return [...acc, current];
      }
    }, [] as Transcription[]);
    setDisplayTranscriptions(mergedSorted);
  }, [rawSegments]);

  return (
    <AgentContext.Provider value={{ displayTranscriptions, agent }}>
      {children}
    </AgentContext.Provider>
  );
}

export function useAgent() {
  const context = useContext(AgentContext);
  if (context === undefined) {
    throw new Error("useAgent must be used within an AgentProvider");
  }
  return context;
}
