import { KeyRound, Mic, ShieldCheck, Square, Waves } from "lucide-react";
import ExamIntro from "@/components/exam/ExamIntro";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import type { Mode } from "@/lib/api";

interface Props {
  mode: Mode;
  connected: boolean;
  /** No OpenAI key yet: nothing can start until one is added. */
  needsKey: boolean;
  onStart: () => void;
  onOpenSettings: () => void;
}

const TIPS = [
  { icon: Waves, text: "Speak naturally. A short pause won't end your turn." },
  { icon: Square, text: "Stop her any time with the Stop button or the Esc key." },
  { icon: ShieldCheck, text: "Your voice is processed on this computer; only the text of the chat goes to the AI model." },
];

/** What you see when you open a new conversation or a new mock test. */
export default function StartView({ mode, connected, needsKey, onStart, onOpenSettings }: Props) {
  const keyNotice = needsKey && (
    <Alert variant="warning" className="mx-auto max-w-3xl items-center">
      <KeyRound />
      <AlertTitle>Add your OpenAI API key to get started</AlertTitle>
      <AlertDescription>
        <p>Jessica uses it to write her replies. It's saved on this computer only.</p>
      </AlertDescription>
      <Button size="sm" onClick={onOpenSettings} className="col-start-2 mt-2 justify-self-start">
        Add key
      </Button>
    </Alert>
  );

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto p-7">
      {keyNotice}
      {mode === "ielts" ? (
        <ExamIntro connected={connected && !needsKey} onBegin={onStart} />
      ) : (
        <Card className="mx-auto w-full max-w-3xl">
          <CardHeader>
            <CardTitle className="text-xl">New conversation</CardTitle>
            <CardDescription className="text-[15px] leading-relaxed">
              Talk to Jessica out loud. She listens, answers in her own voice, and keeps everything in your history so
              you can come back to it.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ul className="grid gap-3">
              {TIPS.map(({ icon: Icon, text }) => (
                <li key={text} className="flex items-start gap-3 text-sm text-muted-foreground">
                  <Icon className="mt-0.5 size-4 shrink-0" />
                  {text}
                </li>
              ))}
            </ul>
          </CardContent>
          <CardFooter>
            <Button size="lg" onClick={onStart} disabled={!connected || needsKey}>
              <Mic className="size-4" /> {connected ? "Start talking" : "Connecting…"}
            </Button>
          </CardFooter>
        </Card>
      )}
    </div>
  );
}
