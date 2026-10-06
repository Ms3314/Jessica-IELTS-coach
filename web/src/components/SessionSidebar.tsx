import { useState } from "react";
import type { LucideIcon } from "lucide-react";
import {
  Bot,
  Brain,
  Cloud,
  Compass,
  Crown,
  Diamond,
  Feather,
  Flame,
  Gem,
  Ghost,
  Globe,
  Heart,
  Key,
  Leaf,
  Lightbulb,
  Map,
  MessageCircle,
  MessageSquareText,
  Moon,
  MoreVertical,
  Music,
  Palette,
  Pencil,
  Plus,
  Rocket,
  Shield,
  Shuffle,
  Smile,
  Snowflake,
  Sparkles,
  Star,
  Sun,
  Target,
  Trash2,
  Umbrella,
  Wallet,
  Watch,
  Zap,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

export interface Session {
  id: string;
  name: string;
  color: string;
  icon: string;
  summary?: string | null;
  updated_at: number;
}

interface Props {
  sessions: Session[];
  activeId: string;
  open: boolean;
  onSelect: (id: string) => void;
  onNew: () => void;
  onUpdate: (id: string, patch: Partial<Session>) => void;
  onDelete: (id: string) => void;
}

const ICONS: Record<string, LucideIcon> = {
  Bot,
  Sparkles,
  Zap,
  MessageCircle,
  Brain,
  Lightbulb,
  Star,
  Heart,
  Moon,
  Sun,
  Cloud,
  Flame,
  Gem,
  Rocket,
  Target,
  Compass,
  Crown,
  Diamond,
  Feather,
  Ghost,
  Globe,
  Key,
  Leaf,
  Map,
  Music,
  Palette,
  Shield,
  Smile,
  Snowflake,
  Umbrella,
  Wallet,
  Watch,
};

const ICON_NAMES = Object.keys(ICONS);

function SessionIcon({ name }: { name: string }) {
  const Icon = ICONS[name] || Bot;
  return <Icon className="size-4" />;
}

function randomIcon(current: string): string {
  const others = ICON_NAMES.filter((n) => n !== current);
  return others[Math.floor(Math.random() * others.length)];
}

export default function SessionSidebar({
  sessions,
  activeId,
  open,
  onSelect,
  onNew,
  onUpdate,
  onDelete,
}: Props) {
  const [summaryId, setSummaryId] = useState<string | null>(null);
  const [summaryText, setSummaryText] = useState<string | null>(null);
  const [editingId, setEditingId] = useState<string | null>(null);

  const summarize = async (id: string) => {
    setSummaryId(id);
    setSummaryText("Summarising…");
    try {
      const res = await fetch(`/sessions/${id}/summarize`, { method: "POST" });
      const data = await res.json();
      setSummaryText(data.summary);
      // Tell parent to refresh the list so the cached summary appears.
      onUpdate(id, {});
    } catch {
      setSummaryText("Could not generate summary.");
    }
  };

  return (
    <>
      <aside className={`session-sidebar ${open ? "" : "closed"}`}>
        <div className="sidebar-header">
          <h2>Sessions</h2>
        </div>

        <ScrollArea className="sidebar-list">
          {sessions.map((s) => (
            <div
              key={s.id}
              className={`session-item ${s.id === activeId ? "active" : ""}`}
              onClick={() => onSelect(s.id)}
            >
              <button
                type="button"
                className="session-icon"
                style={{ color: s.color }}
                onClick={(e) => {
                  e.stopPropagation();
                  onUpdate(s.id, { icon: randomIcon(s.icon) });
                }}
                title="Change icon"
              >
                <SessionIcon name={s.icon} />
              </button>

              {editingId === s.id ? (
                <Input
                  className="session-name-input"
                  value={s.name}
                  autoFocus
                  onClick={(e) => e.stopPropagation()}
                  onChange={(e) => onUpdate(s.id, { name: e.target.value })}
                  onBlur={() => setEditingId(null)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") setEditingId(null);
                  }}
                />
              ) : (
                <span
                  className="session-name"
                  onDoubleClick={() => setEditingId(s.id)}
                  title={s.id}
                >
                  {s.name}
                </span>
              )}

              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon-xs"
                    className="session-menu"
                    onClick={(e) => e.stopPropagation()}
                  >
                    <MoreVertical className="size-3.5" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  <DropdownMenuItem onClick={() => setEditingId(s.id)}>
                    <Pencil className="size-3.5 mr-2" />
                    Rename
                  </DropdownMenuItem>
                  <DropdownMenuItem
                    onClick={() => onUpdate(s.id, { icon: randomIcon(s.icon) })}
                  >
                    <Shuffle className="size-3.5 mr-2" />
                    Change icon
                  </DropdownMenuItem>
                  <Dialog>
                    <DialogTrigger asChild>
                      <DropdownMenuItem
                        onSelect={(e) => e.preventDefault()}
                        onClick={() => summarize(s.id)}
                      >
                        <MessageSquareText className="size-3.5 mr-2" />
                        Summarise
                      </DropdownMenuItem>
                    </DialogTrigger>
                    <DialogContent>
                      <DialogHeader>
                        <DialogTitle>{s.name}</DialogTitle>
                      </DialogHeader>
                      <p className="text-sm leading-relaxed text-muted-foreground">
                        {summaryId === s.id ? summaryText : s.summary || "No summary yet."}
                      </p>
                    </DialogContent>
                  </Dialog>
                  <DropdownMenuItem
                    className="text-destructive focus:text-destructive"
                    onClick={() => onDelete(s.id)}
                  >
                    <Trash2 className="size-3.5 mr-2" />
                    Delete
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
          ))}
        </ScrollArea>

        <Button variant="outline" className="w-full gap-2" onClick={onNew}>
          <Plus className="size-4" />
          New session
        </Button>
      </aside>

      {/* Summary dialog state is lifted; we render a controlled dialog here so
          the async summary can update after fetch. */}
    </>
  );
}
