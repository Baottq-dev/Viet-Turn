# run_pipeline.ps1 - Chay toan bo pipeline (Windows PowerShell)
$ErrorActionPreference = "Stop"
chcp 65001 | Out-Null

$AUDIO_RAW = "data/audio"
$AUDIO_DIR = "data/audio_split"
$RTTM_DIR = "data/rttm"
$VA_DIR = "data/va_matrices"
$LABEL_DIR = "data/vap_labels"
$TRANSCRIPT_DIR = "data/transcripts"
$TEXT_DIR = "data/text_frames"
$OUTPUT_DIR = "data"

# Write-Host "=== Step 0: Download Audio ==="
# python scripts/00_download_audio.py --output $AUDIO_RAW
# if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Write-Host "=== Step 0b: Split Long Audio ==="
# python scripts/00b_split_audio.py --input $AUDIO_RAW --output $AUDIO_DIR --segment-min 10
# if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Write-Host "=== Step 1: Diarization ==="
# python scripts/01_diarize.py --input $AUDIO_DIR --output $RTTM_DIR --num-speakers 2
# if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Write-Host "=== Step 2: VA Matrix ==="
# python scripts/02_build_va_matrix.py --rttm-dir $RTTM_DIR --audio-dir $AUDIO_DIR --output $VA_DIR
# if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Write-Host "=== Step 3: VAP Labels ==="
# python scripts/03_generate_labels.py --input $VA_DIR --output $LABEL_DIR
# if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Write-Host "=== Step 4: Transcription ==="
# python scripts/04_transcribe.py --input $AUDIO_DIR --output $TRANSCRIPT_DIR --model large-v3
# if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== Step 5: Text Alignment ==="
python scripts/05_align_text.py --transcripts $TRANSCRIPT_DIR --va-matrices $VA_DIR --output $TEXT_DIR
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== Step 6: Create Manifest ==="
python scripts/06_create_manifest.py --audio-dir $AUDIO_DIR --va-dir $VA_DIR --label-dir $LABEL_DIR --text-dir $TEXT_DIR --output $OUTPUT_DIR
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== Step 7: Validate ==="
python scripts/07_validate_data.py --manifest "$OUTPUT_DIR/vap_manifest_train.json"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== Pipeline hoan tat! ==="
