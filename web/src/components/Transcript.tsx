import { useEffect, useRef } from "react";
import { Badge } from "@/components/ui/badge";
import { cn } from "cn";

export interface TranscriptLine {
  id?: number | string;
  role: "user" | "assistant";
  text: string;
  interrupted?: boolean;
}

interface Props {
  lines: TranscriptLine[];
  /** Keep the newest line in view (for the live call). */
  follow?: boolean;
  assistantName?: string;
}

/** A plain, readable conversation: who said what, in order. */
export default function Transcript({ lines, follow = false, assistantName = "Jessica" }: Props) {
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (follow) endRef.current?.scrollIntoView({ block: "end", behavior: "smooth" });
  }, [follow, lines]);

  return (
    <div className="flex flex-col gap-5" role="log" aria-live={follow ? "polite" : "off"}>
      {lines.map((l, i) => {
        const user = l.role === "user";
        return (
          <article
            key={l.id ?? i}
            className={cn("flex max-w-full animate-[pop_0.15s_ease-out] flex-col gap-1", user && "items-end")}
          >
            <header className="text-xs font-medium text-muted-foreground">{user ? "You" : assistantName}</header>
            <p
              className={cn(
                "text-[15px] leading-relaxed whitespace-pre-wrap",
                user && "max-w-[85%] rounded-2xl rounded-tr-md bg-secondary px-4 py-2.5 text-left"
              )}
            >
              {l.text || <span className="text-muted-foreground italic">(cut off before she spoke)</span>}
              {l.interrupted && l.text && (
                <Badge variant="outline" className="ml-2 align-middle font-normal text-muted-foreground">
                  interrupted
                </Badge>
              )}
            </p>
          </article>
        );
      })}
      <div ref={endRef} />
    </div>
  );
}
