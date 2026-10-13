import { Loader2 } from "lucide-react";

interface Props {
  label: string;
  hint?: string;
}

/** A round loading ring with a label: shown until Jessica's first words reach the speakers. */
export default function Loader({ label, hint }: Props) {
  return (
    <div className="flex flex-col items-center gap-2 text-center" role="status" aria-live="polite">
      <Loader2 className="size-9 animate-spin text-muted-foreground" aria-hidden="true" />
      <p className="mt-2 text-[15px] font-medium">{label}</p>
      {hint && <p className="text-sm text-muted-foreground">{hint}</p>}
    </div>
  );
}
