#!/usr/bin/env bash
# D1 / Small CNN Zoo STL10 - download verificabile per Model X-Ray.
# Fonte: https://zenodo.org/records/6631784 (DOI: 10.5281/zenodo.6631784)
# Eseguire con: bash scripts/download_d1_stl10.sh [--seed-only|--all]
# I dati restano fuori dalla repository GitHub.
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Uso: bash scripts/download_d1_stl10.sh [--all|--seed-only]

  --all         Scarica ed estrae i tre archivi STL10 (predefinito).
  --seed-only   Scarica ed estrae solo stl_small_seed.zip (~767 MB), per prova.

Percorso predefinito: $HOME/mlsec-local/datasets/ghrp_zoos
Per cambiarlo, imposta MODELXRAY_GHRP_DIR prima dell'avvio.
EOF
}

case "${1:---all}" in
  --all)        ARCHIVES=(stl_small_hyp_fix stl_small_hyp_rand stl_small_seed) ;;
  --seed-only)  ARCHIVES=(stl_small_seed) ;;
  -h|--help)    usage; exit 0 ;;
  *)            usage >&2; exit 2 ;;
esac
if (( $# > 1 )); then usage >&2; exit 2; fi

for cmd in curl unzip md5sum python3 df; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "ERRORE: comando mancante: $cmd" >&2
    echo "Su Ubuntu: sudo apt install -y curl unzip python3 coreutils" >&2
    exit 1
  fi
done

DATA_DIR="${MODELXRAY_GHRP_DIR:-$HOME/mlsec-local/datasets/ghrp_zoos}"
mkdir -p "$DATA_DIR"
DATA_DIR="$(cd "$DATA_DIR" && pwd -P)"
BASE_URL='https://zenodo.org/records/6631784/files'

# MD5 pubblicati nella scheda ufficiale del record Zenodo 6631784.
declare -A MD5=(
  [stl_small_hyp_fix]='5cd2888b7c19f3d5ce77667b23c1f0fd'
  [stl_small_hyp_rand]='bb4db0f5f4d278fbcc2f0194a7e3de3b'
  [stl_small_seed]='6b81b38ffdf767f317210ab3f68d9753'
)

checksum_ok() {
  local file="$1" expected="$2"
  [[ -s "$file" ]] && printf '%s  %s\n' "$expected" "$file" | md5sum --check --status
}

is_zip_prefix() {
  local prefix
  [[ -s "$1" ]] || return 1
  prefix="$(head -c 2 "$1")"
  [[ "$prefix" == 'PK' ]]
}

# Esclude path traversal e link simbolici negli ZIP prima dell'estrazione.
check_zip_entries() {
  python3 - "$1" <<'PY'
import pathlib
import stat
import sys
import zipfile

archive = sys.argv[1]
with zipfile.ZipFile(archive) as z:
    for item in z.infolist():
        name = item.filename.replace('\\', '/')
        path = pathlib.PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or not path.parts:
            raise SystemExit(f'Archivio non sicuro, percorso inatteso: {name!r}')
        mode = item.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise SystemExit(f'Archivio non sicuro, link simbolico: {name!r}')
print('Struttura ZIP: controllo percorsi OK')
PY
}

echo '=================================================='
echo 'Model X-Ray - D1 Small CNN Zoo (STL10)'
echo "Destinazione: $DATA_DIR"
echo 'Sorgente: https://zenodo.org/records/6631784'
echo '=================================================='
df -h "$DATA_DIR"

for name in "${ARCHIVES[@]}"; do
  zip_path="$DATA_DIR/${name}.zip"
  partial_path="${zip_path}.part"
  marker="$DATA_DIR/.${name}.extracted.ok"
  expected="${MD5[$name]}"

  echo
  echo "======== $name ========"

  if [[ -f "$marker" ]]; then
    echo 'Estrazione gia completata in una esecuzione precedente: salto.'
    continue
  fi

  if checksum_ok "$zip_path" "$expected"; then
    echo 'Archivio ZIP gia presente e checksum MD5 valido.'
  else
    if [[ -e "$zip_path" ]]; then
      # Il vecchio comando copiato da Markdown puo avere creato un file HTML.
      if [[ ! -e "$partial_path" ]] && is_zip_prefix "$zip_path"; then
        mv -- "$zip_path" "$partial_path"
        echo 'ZIP incompleto: provo a riprendere il download.'
      else
        backup="${zip_path}.nonvalido.$(date +%Y%m%d_%H%M%S)"
        mv -- "$zip_path" "$backup"
        echo "Archivio non valido conservato in: $backup"
      fi
    fi

    if [[ -f "$partial_path" ]] && ! is_zip_prefix "$partial_path"; then
      backup="${partial_path}.nonvalido.$(date +%Y%m%d_%H%M%S)"
      mv -- "$partial_path" "$backup"
      echo "Download parziale non ZIP conservato in: $backup"
    fi

    echo "Scarico ${name}.zip da Zenodo (ripresa automatica se possibile)..."
    curl --fail --location --retry 5 --retry-delay 5 \
      --connect-timeout 30 --continue-at - \
      --output "$partial_path" \
      "$BASE_URL/${name}.zip?download=1"

    echo 'Controllo checksum MD5 pubblicato da Zenodo...'
    if ! checksum_ok "$partial_path" "$expected"; then
      echo "ERRORE: checksum non corrispondente per $partial_path" >&2
      echo 'NON estraggo il file. Controlla il download e riprova.' >&2
      echo 'Se necessario, rinomina il .part non valido e riavvia lo script.' >&2
      exit 1
    fi
    mv -- "$partial_path" "$zip_path"
  fi

  check_zip_entries "$zip_path"
  echo 'Verifico la struttura degli archivi (prime voci):'
  unzip -Z -1 "$zip_path" | sed -n '1,6p'

  echo "Estraggo $name in $DATA_DIR ..."
  # -o sovrascrive eventuali file incompleti dopo interruzioni precedenti.
  # Non creare il marker fino a estrazione completamente conclusa.
  unzip -o -q "$zip_path" -d "$DATA_DIR"
  printf 'Zenodo: 10.5281/zenodo.6631784\nArchivio: %s.zip\nMD5: %s\n' \
    "$name" "$expected" > "$marker"
  echo "Estrazione completata: $name"
done

echo
echo '=================================================='
echo 'Download/estrazione conclusi per gli archivi richiesti.'
echo "MODELXRAY_GHRP_DIR=$DATA_DIR"
echo 'Spazio occupato:'
du -sh "$DATA_DIR"
echo 'Cartelle principali:'
find "$DATA_DIR" -mindepth 1 -maxdepth 2 -type d | sed -n '1,25p'
echo 'Esempi di dataset.pt disponibili:'
find "$DATA_DIR" -name dataset.pt -type f -print -quit
echo
 echo 'IMPORTANTE: per il prossimo comando imposta MODELXRAY_GHRP_DIR nella shell,'
echo 'oppure passala con env. I file .pt devono essere caricati solo dopo'
echo 'aver verificato provenienza e modalita di deserializzazione.'
