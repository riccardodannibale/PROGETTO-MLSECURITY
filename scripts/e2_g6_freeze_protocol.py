#!/usr/bin/env python3
"""Freeze reviewed G4 for validation-only after explicit Riccardo confirmation.

Does NOT authorize E2 test and NEVER creates a test approval JSON.
"""
from __future__ import annotations
import hashlib
import subprocess
from pathlib import Path
from datetime import date

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "docs/E2_PROTOCOL_FINAL_REVIEW.md"
FROZEN = ROOT / "docs/E2_PROTOCOL_FROZEN.md"
RUNNER = ROOT / "scripts/run_e2_paired_v0.py"
REVIEW_SHA = "c822e5a27146b6fc00a568e0b094a777227b542aaf24b6dda3cd4733a7cb4b05"
RUNNER_SHA = "4719e6d947c5c5f7acba09f464ec0d2e5abf5182c642af8a063789ca217849ac"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def main():
    if git("branch", "--show-current") != "ricky":
        raise SystemExit("STOP: expected branch ricky")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("STOP: tracked working tree not clean; inspect git status before freezing")
    if FROZEN.exists():
        raise SystemExit(f"STOP: freeze already exists; do not silently overwrite {FROZEN}")
    for label, path, expected in (("G4 review", REVIEW, REVIEW_SHA), ("E2 runner", RUNNER, RUNNER_SHA)):
        if not path.is_file() or sha(path) != expected:
            raise SystemExit(f"STOP: {label} not at reviewed SHA {expected}")
    s = REVIEW.read_text(encoding="utf-8")
    old_status = "**Stato: REVIEW — NON È ANCORA FROZEN.** Nessun test E2 né training E2 autorizzato da questo file."
    new_status = ("**Stato: FROZEN PER VALIDATION E2 — 22 ottobre 2026.** "
                  "Autorizzata esclusivamente la campagna ufficiale di dieci fit di validation X-LSB. "
                  "**TEST ID/OOD NON AUTORIZZATO**: richiede audit 10/10 e nuovo atto congiunto. "
                  "Il precedente singolo fit di validation era una diagnostica, non un risultato ufficiale.")
    if s.count(old_status) != 1 or s.count("## 7. Approvazioni") != 1:
        raise SystemExit("STOP: unexpected G4 review format")
    s = s.replace(old_status, new_status, 1)
    old_note = ("Fabrizio ha confermato gli audit cross-platform e revisionato l'architettura. "
                "**La revisione del runner G3-C e di QUESTO protocollo richiede ancora firma esplicita**.")
    new_note = ("Fabrizio ha confermato gli audit cross-platform e revisionato l'architettura. "
                "Ha inoltre approvato esplicitamente senza modifiche il protocollo G4 e i capisaldi E2 "
                "(messaggio riportato da Riccardo il 22 ottobre 2026). "
                "La revisione separata del codice G3-C non è documentata da quel messaggio: "
                "il pilot utility resta esplorativo e non viene trattato come risultato principale E2.")
    if old_note in s:
        s = s.replace(old_note, new_note, 1)
    s = s[:s.index("## 7. Approvazioni")] + f"""## 7. Freeze, approvazioni e ambito

- [x] **Riccardo:** approvazione esplicita del presente freeze **solo validation**, data {date.today().isoformat()}, confermata digitando `APPROVO` nel comando di congelamento.
- [x] **Fabrizio:** approvazione integrale, senza modifiche, di `docs/E2_PROTOCOL_FINAL_REVIEW.md` e di tutti i capisaldi E2, comunicata a Riccardo il 22 ottobre 2026 (fonte: messaggio condiviso da Riccardo; non una firma digitale).
- [x] **Runner G5:** SHA-256 `{RUNNER_SHA}`; sintassi e preflight superati su Ubuntu, test unitario e revisione su Mac; un fit diagnostico di validation Ubuntu superato con QA.
- [x] **Documento sorgente G4:** SHA-256 `{REVIEW_SHA}`.
- [ ] **Autorizzazione al TEST:** NON concessa. Sarà oggetto di una revisione distinta dopo 10/10 fit e audit. Nessun file `approved_for_e2_test` deve essere creato ora.

Il **commit del freeze** sarà identificato dal log Git dopo il commit, per evitare un riferimento circolare al commit che contiene questo stesso documento. **Non modificare il presente file per registrare il suo commit**: conservarne invece l'hash e il commit in un registro separato.

### Vincolo di immutabilità

Seed, attacchi, alpha, metrica primaria, sorgenti, split, runner e condizioni di test sono fissati. Qualsiasi correzione dopo il freeze richiede un *amendment* esplicito e documentato, prima di aprire il test.
"""
    answer = input("Per approvare E2 validation (NON il test), digita esattamente APPROVO: ").strip()
    if answer != "APPROVO":
        raise SystemExit("STOP: Riccardo non ha espresso approvazione esplicita")
    # Normalize trailing Markdown hard-break spaces so git diff --check stays clean.
    s = "\n".join(line.rstrip() for line in s.splitlines()) + "\n"
    FROZEN.write_text(s, encoding="utf-8")
    print("E2 G6 FREEZE DRAFT CREATED (requires Git commit before validation)")
    print("Frozen protocol SHA-256:", sha(FROZEN))
    print("E2 runner SHA-256:", sha(RUNNER))
    print("TEST E2: LOCKED")


if __name__ == "__main__":
    main()
