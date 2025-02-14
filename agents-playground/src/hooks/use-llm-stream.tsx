import { useDataChannel } from "@livekit/components-react";
import { useState, useCallback, useEffect, useRef } from "react";

interface LLMResponse {
  type: "llm_response";
  text: string;
  timestamp: number;
}

export function useLLMStream() {
  const [llmResponses, setLLMResponses] = useState<LLMResponse[]>([]);
  const currentMessageRef = useRef<{text: string, timestamp: number} | null>(null);

  const onDataReceived = useCallback((msg: any) => {
    if (msg.topic === "llm_stream") {
      const decoded = JSON.parse(new TextDecoder("utf-8").decode(msg.payload));
      if (decoded.type === "llm_response") {
        setLLMResponses((prev) => {
          // If there's no current message or it's been more than 2 seconds since the last token,
          // start a new message
          if (!currentMessageRef.current || 
              decoded.timestamp - currentMessageRef.current.timestamp > 2000) {
            currentMessageRef.current = {
              text: decoded.text,
              timestamp: decoded.timestamp
            };
            return [...prev, decoded];
          }

          // Otherwise, update the last message with accumulated text
          const updatedResponses = [...prev];
          const lastResponse = updatedResponses[updatedResponses.length - 1];
          currentMessageRef.current.text += decoded.text;
          currentMessageRef.current.timestamp = decoded.timestamp;
          
          updatedResponses[updatedResponses.length - 1] = {
            ...lastResponse,
            text: currentMessageRef.current.text,
            timestamp: decoded.timestamp
          };

          return updatedResponses;
        });
      }
    }
  }, []);

  useDataChannel('llm_stream', onDataReceived);

  return { llmResponses };
}
