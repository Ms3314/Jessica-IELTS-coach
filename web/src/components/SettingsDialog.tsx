import { Settings as SettingsIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Switch } from "@/components/ui/switch";
import type { Settings } from "@/lib/settings";

interface Props {
  settings: Settings;
  onChange: (patch: Partial<Settings>) => void;
}

interface Row {
  key: keyof Settings;
  title: string;
  description: string;
}

// One entry per setting; add rows here as settings are added.
const ROWS: Row[] = [
  {
    key: "voiceInterrupt",
    title: "Interrupt by talking",
    description:
      "Cut Jessica off just by speaking over her. Loud rooms can trigger this by accident, so it is off by default. The Stop button (or Esc) always works.",
  },
];

export default function SettingsDialog({ settings, onChange }: Props) {
  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button variant="ghost" size="icon-sm" aria-label="Settings">
          <SettingsIcon className="size-5" />
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Settings</DialogTitle>
          <DialogDescription>Saved in this browser.</DialogDescription>
        </DialogHeader>

        <div className="flex flex-col gap-4">
          {ROWS.map((row) => (
            <div key={row.key} className="flex items-start justify-between gap-4">
              <label htmlFor={`setting-${row.key}`} className="flex flex-col gap-1">
                <span className="text-sm font-medium">{row.title}</span>
                <span className="text-xs text-muted-foreground">{row.description}</span>
              </label>
              <Switch
                id={`setting-${row.key}`}
                checked={Boolean(settings[row.key])}
                onCheckedChange={(checked) => onChange({ [row.key]: checked })}
              />
            </div>
          ))}
        </div>
      </DialogContent>
    </Dialog>
  );
}
