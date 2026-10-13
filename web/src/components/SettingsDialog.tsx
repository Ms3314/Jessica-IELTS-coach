import { useState } from "react";
import { Check, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { Switch } from "@/components/ui/switch";
import { saveKeys, type KeyStatus, type ServerSettings } from "@/lib/api";
import { PAUSE_RANGE, type Settings } from "@/lib/settings";

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  settings: Settings;
  onChange: (patch: Partial<Settings>) => void;
  server: ServerSettings | null;
  /** Called after a key is saved, so the app can refresh what it knows. */
  onKeysSaved: () => void;
}

/** One masked API-key box: shows whether a key is set, never the key itself. */
function KeyField({
  id,
  label,
  help,
  field,
  status,
  placeholder,
  onSaved,
}: {
  id: string;
  label: string;
  help: string;
  field: "openai_api_key" | "tavily_api_key";
  status: KeyStatus | undefined;
  placeholder: string;
  onSaved: () => void;
}) {
  const [value, setValue] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<{ ok: boolean; text: string } | null>(null);

  const save = async () => {
    if (!value.trim()) return;
    setBusy(true);
    setMessage(null);
    const result = await saveKeys({ [field]: value.trim() });
    setBusy(false);
    if (result.ok) {
      setValue("");
      setMessage({ ok: true, text: result.note ?? "Saved. It's in use now." });
      onSaved();
    } else {
      setMessage({ ok: false, text: result.error ?? "Could not save." });
    }
  };

  return (
    <div className="grid gap-1.5">
      <Label htmlFor={id}>{label}</Label>
      <p className="text-[13px] leading-snug text-muted-foreground">{help}</p>
      <p className="flex items-center gap-1.5 text-[13px] text-muted-foreground">
        {status?.set ? (
          <>
            <Check className="size-3.5 text-emerald-700" /> Key set{" "}
            <code className="rounded bg-muted px-1.5 py-0.5 text-xs">{status.hint}</code>
            {status.source === "environment" && " (from your .env file)"}
          </>
        ) : (
          "No key set"
        )}
      </p>
      <div className="flex gap-2">
        <Input
          id={id}
          type="password"
          autoComplete="off"
          spellCheck={false}
          placeholder={status?.set ? "Paste a new key to replace it" : placeholder}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && save()}
        />
        <Button onClick={save} disabled={busy || !value.trim()}>
          {busy ? <Loader2 className="size-4 animate-spin" /> : "Save"}
        </Button>
      </div>
      {message && (
        <p className={`text-[13px] ${message.ok ? "text-emerald-700" : "text-red-700"}`} role="status">
          {message.text}
        </p>
      )}
    </div>
  );
}

export default function SettingsDialog({ open, onOpenChange, settings, onChange, server, onKeysSaved }: Props) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[88vh] overflow-y-auto sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>Settings</DialogTitle>
          <DialogDescription>Saved on this computer.</DialogDescription>
        </DialogHeader>

        <Separator />
        <section className="grid gap-5">
          <h3 className="text-xs font-semibold tracking-wider text-muted-foreground uppercase">AI model</h3>
          <KeyField
            id="openai-key"
            label="OpenAI API key"
            help={`Used to write Jessica's replies and your test reports${server ? ` (model: ${server.model})` : ""}.`}
            field="openai_api_key"
            status={server?.keys.openai_api_key}
            placeholder="sk-…"
            onSaved={onKeysSaved}
          />
          <KeyField
            id="tavily-key"
            label="Tavily API key (optional)"
            help="Lets Jessica search the web when she needs a fact."
            field="tavily_api_key"
            status={server?.keys.tavily_api_key}
            placeholder="tvly-…"
            onSaved={onKeysSaved}
          />
        </section>

        <Separator />
        <section className="grid gap-5">
          <h3 className="text-xs font-semibold tracking-wider text-muted-foreground uppercase">Voice</h3>
          <div className="grid gap-2">
            <div className="flex items-center justify-between">
              <Label htmlFor="setting-pause">Wait before answering</Label>
              <span className="text-sm text-muted-foreground tabular-nums">{settings.pauseS.toFixed(1)} s</span>
            </div>
            <input
              id="setting-pause"
              type="range"
              min={PAUSE_RANGE.min}
              max={PAUSE_RANGE.max}
              step={PAUSE_RANGE.step}
              value={settings.pauseS}
              onChange={(e) => onChange({ pauseS: Number(e.target.value) })}
              className="w-full accent-primary"
            />
            <p className="text-[13px] leading-snug text-muted-foreground">
              How long you must stay quiet before Jessica decides you're done. Raise it if she answers too soon.
            </p>
          </div>
          <div className="grid gap-1.5">
            <div className="flex items-center justify-between gap-3">
              <Label htmlFor="setting-interrupt">Interrupt by talking</Label>
              <Switch
                id="setting-interrupt"
                checked={settings.voiceInterrupt}
                onCheckedChange={(checked) => onChange({ voiceInterrupt: checked })}
              />
            </div>
            <p className="text-[13px] leading-snug text-muted-foreground">
              Cut Jessica off just by speaking over her. Noisy rooms can trigger it by accident. The Stop button always
              works.
            </p>
          </div>
        </section>

        <Separator />
        <section className="grid gap-2">
          <h3 className="text-xs font-semibold tracking-wider text-muted-foreground uppercase">Privacy</h3>
          <p className="text-[13px] leading-relaxed text-muted-foreground">
            Speech recognition and Jessica's voice run on this computer. Only the text of the conversation is sent to
            the AI model. Mock-test recordings stay here and are deleted with their session.
          </p>
        </section>
      </DialogContent>
    </Dialog>
  );
}
