import { useMemo, useState } from "react";
import { FileText, MessageSquare, MoreHorizontal, Pencil, Plus, Search, Settings, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { cn } from "cn";
import type { SessionMeta } from "@/lib/api";
import { groupByDay, isTestSession, matchesSearch, timeLabel } from "@/lib/history";

interface Props {
  sessions: SessionMeta[];
  activeId: string;
  /** While a call is running, other items can't be opened. */
  locked: boolean;
  onNewChat: () => void;
  onNewTest: () => void;
  onOpen: (session: SessionMeta) => void;
  onRename: (id: string, name: string) => void;
  onDelete: (session: SessionMeta) => void;
  onOpenSettings: () => void;
}

/** The left column: start something new, or reopen anything you've done before. */
export default function HistorySidebar({
  sessions,
  activeId,
  locked,
  onNewChat,
  onNewTest,
  onOpen,
  onRename,
  onDelete,
  onOpenSettings,
}: Props) {
  const [query, setQuery] = useState("");
  const [editingId, setEditingId] = useState<string | null>(null);

  const groups = useMemo(() => groupByDay(sessions.filter((s) => matchesSearch(s, query))), [sessions, query]);

  const commitRename = (id: string, value: string) => {
    setEditingId(null);
    const name = value.trim();
    if (name) onRename(id, name);
  };

  return (
    <aside className="flex min-h-0 flex-col border-r bg-muted/40" aria-label="History">
      <div className="px-5 pt-5 pb-3">
        <span className="text-[17px] font-semibold tracking-tight">Jessica</span>
      </div>

      <div className="grid gap-2 px-3 pb-3">
        <Button size="sm" onClick={onNewChat} disabled={locked} className="justify-start">
          <Plus className="size-4" /> New conversation
        </Button>
        <Button size="sm" variant="outline" onClick={onNewTest} disabled={locked} className="justify-start bg-background">
          <FileText className="size-4" /> New IELTS test
        </Button>
      </div>

      <div className="relative px-3 pb-2">
        <Search className="pointer-events-none absolute top-1/2 left-5.5 size-4 -translate-y-1/2 text-muted-foreground" />
        <Input
          type="search"
          placeholder="Search history"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          aria-label="Search history"
          className="h-9 bg-background pl-8"
        />
      </div>

      <ScrollArea className="min-h-0 flex-1">
        <nav className="px-2 pb-3" aria-label="Past sessions">
          {groups.length === 0 && (
            <p className="px-3 py-4 text-sm text-muted-foreground">
              {query ? "Nothing matches your search." : "Your conversations and tests will appear here."}
            </p>
          )}
          {groups.map((g) => (
            <section key={g.label} className="mt-3 first:mt-1">
              <h2 className="px-3 pb-1 text-[11px] font-semibold tracking-wider text-muted-foreground uppercase">
                {g.label}
              </h2>
              <ul className="grid gap-0.5">
                {g.items.map((s) => {
                  const Icon = isTestSession(s) ? FileText : MessageSquare;
                  const active = s.id === activeId;
                  return (
                    <li
                      key={s.id}
                      className={cn(
                        "group relative rounded-lg transition-colors hover:bg-accent",
                        active && "bg-accent"
                      )}
                    >
                      {editingId === s.id ? (
                        <Input
                          autoFocus
                          defaultValue={s.name}
                          aria-label="Rename"
                          className="m-1 h-8 w-[calc(100%-0.5rem)] bg-background"
                          onBlur={(e) => commitRename(s.id, e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === "Enter") commitRename(s.id, e.currentTarget.value);
                            if (e.key === "Escape") setEditingId(null);
                          }}
                        />
                      ) : (
                        <button
                          type="button"
                          onClick={() => onOpen(s)}
                          disabled={locked && !active}
                          aria-current={active ? "page" : undefined}
                          className="grid w-full grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-2.5 rounded-lg px-3 py-2 pr-9 text-left text-sm outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
                        >
                          <Icon className="size-4 text-muted-foreground" aria-hidden="true" />
                          <span className="truncate">{s.name}</span>
                          <span className="text-xs text-muted-foreground group-hover:invisible">
                            {timeLabel(s.updated_at)}
                          </span>
                        </button>
                      )}
                      {editingId !== s.id && (
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <Button
                              variant="ghost"
                              size="icon-xs"
                              aria-label={`Options for ${s.name}`}
                              disabled={locked && active}
                              className="absolute top-1/2 right-1.5 hidden -translate-y-1/2 group-hover:inline-flex data-[state=open]:inline-flex"
                            >
                              <MoreHorizontal />
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="end">
                            <DropdownMenuItem onSelect={() => setEditingId(s.id)}>
                              <Pencil /> Rename
                            </DropdownMenuItem>
                            <DropdownMenuItem variant="destructive" onSelect={() => onDelete(s)}>
                              <Trash2 /> Delete
                            </DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      )}
                    </li>
                  );
                })}
              </ul>
            </section>
          ))}
        </nav>
      </ScrollArea>

      <Separator />
      <div className="p-3">
        <Button variant="ghost" size="sm" onClick={onOpenSettings} className="w-full justify-start text-muted-foreground">
          <Settings className="size-4" /> Settings
        </Button>
      </div>
    </aside>
  );
}
