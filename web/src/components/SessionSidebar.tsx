import { useState } from "react";
import { MessageSquareText, MoreVertical, PanelLeft, Plus, Pencil, Trash2 } from "lucide-react";
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
  summary?: string | null;
  updated_at: number;
}

interface Props {
  sessions: Session[];
  activeId: string;
  open: boolean;
  onToggle: () => void;
  onSelect: (id: string) => void;
  onNew: () => void;
  onUpdate: (id: string, patch: Partial<Session>) => void;
  onDelete: (id: string) => void;
}

export default function SessionSidebar({
  sessions,
  activeId,
  open,
  onToggle,
  onSelect,
  onNew,
  onUpdate,
  onDelete,
}: Props) {
  const [summaryId, setSummaryId] = useState<string | null>(null);
  const [summaryText, setSummaryText] = useState<string | null>(null);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [generating, setGenerating] = useState(false);

  const activeSession = sessions.find((s) => s.id === activeId);

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

  const summarizeActive = async () => {
    if (!activeId) return;
    setGenerating(true);
    try {
      await fetch(`/sessions/${activeId}/summarize`, { method: "POST" });
      onUpdate(activeId, {});
    } finally {
      setGenerating(false);
    }
  };

  return (
    <>
      <aside className={`session-sidebar ${open ? "" : "closed"}`}>
        <div className="sidebar-header">
          <h2>Sessions</h2>
          <Button variant="ghost" size="icon-xs" onClick={onToggle} aria-label="Close sidebar">
            <PanelLeft className="size-4" />
          </Button>
        </div>

        <div className="sidebar-summary">
          <div className="sidebar-summary-head">
            <span>Summary</span>
            {!activeSession?.summary && (
              <Button
                size="xs"
                variant="ghost"
                disabled={generating}
                onClick={summarizeActive}
              >
                {generating ? "Generating…" : "Generate"}
              </Button>
            )}
          </div>
          <p className="sidebar-summary-text">
            {activeSession?.summary ||
              "No summary yet — generate one to see what you've talked about."}
          </p>
        </div>

        <ScrollArea className="sidebar-list">
          {sessions.map((s) => (
            <div
              key={s.id}
              className={`session-item ${s.id === activeId ? "active" : ""}`}
              onClick={() => onSelect(s.id)}
            >
              <input
                type="color"
                className="session-color"
                value={s.color}
                onClick={(e) => e.stopPropagation()}
                onChange={(e) => onUpdate(s.id, { color: e.target.value })}
                title="Change colour"
              />

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
