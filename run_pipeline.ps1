# run_pipeline.ps1 - Full pipeline with quality checkpoints
# Chay tung phase, kiem tra chat luong giua cac buoc
#
# Usage:
#   .\run_pipeline.ps1                    # Chay tat ca tu dau
#   .\run_pipeline.ps1 -StartFrom 1       # Chay tu Step 0b (da co audio)
#   .\run_pipeline.ps1 -StartFrom 2       # Chay tu Step 1 (da co split)
#   .\run_pipeline.ps1 -StopAfter 0       # Chi download audio roi dung
#   .\run_pipeline.ps1 -StartFrom 0 -StopAfter 0  # Chi download audio

param(
    [int]$StartFrom = 0,
    [int]$StopAfter = 99
)

$ErrorActionPreference = "Continue"
chcp 65001 | Out-Null
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"

$AUDIO_RAW = "data/audio"
$AUDIO_DIR = "data/audio_split"
$RTTM_DIR = "data/rttm"
$VA_DIR = "data/va_matrices"
$LABEL_DIR = "data/vap_labels"
$TRANSCRIPT_DIR = "data/transcripts"
$TEXT_DIR = "data/text_frames"
$OUTPUT_DIR = "data"
$LOG_DIR = "data/_logs"

# Tao log directory
New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null

function Write-StepHeader($step, $name) {
    $line = "=" * 60
    Write-Host "`n$line" -ForegroundColor Cyan
    Write-Host "  STEP $step : $name" -ForegroundColor Cyan
    Write-Host "$line" -ForegroundColor Cyan
}

function Write-QualityCheck($msg, $status) {
    if ($status -eq "OK") {
        Write-Host "  [OK] $msg" -ForegroundColor Green
    } elseif ($status -eq "WARN") {
        Write-Host "  [WARN] $msg" -ForegroundColor Yellow
    } else {
        Write-Host "  [FAIL] $msg" -ForegroundColor Red
    }
}

# Helper: run python script, show output on console, save copy to log file
function Run-PythonStep($scriptArgs, $logFile) {
    # Use cmd /c to avoid PowerShell stderr handling issues
    $fullCmd = "python $scriptArgs"
    cmd /c "$fullCmd > `"$logFile`" 2>&1"
    $script:_lastPythonExit = $LASTEXITCODE
    # Show log on console (pipe to Out-Host so it doesn't pollute return value)
    if (Test-Path $logFile) { Get-Content $logFile -Encoding UTF8 | Out-Host }
}

# =========================================================================
# STEP 0: Download Audio
# =========================================================================
if ($StartFrom -le 0 -and $StopAfter -ge 0) {
    Write-StepHeader 0 "Download Audio from YouTube"
    Write-Host "  Downloading 65 videos to $AUDIO_RAW ..."
    Write-Host "  (This will take several hours depending on network speed)"
    Write-Host ""

    Run-PythonStep "scripts/00_download_audio.py --output $AUDIO_RAW" "$LOG_DIR/step0_download.log"
    if ($script:_lastPythonExit -ne 0) { Write-Host "STEP 0 FAILED" -ForegroundColor Red; exit 1 }

    # Quality check: count downloaded files
    $audioFiles = @(Get-ChildItem "$AUDIO_RAW/*.wav" -ErrorAction SilentlyContinue).Count
    Write-Host "`n  --- Quality Check: Download ---"
    Write-QualityCheck "$audioFiles WAV files downloaded" $(if ($audioFiles -ge 60) {"OK"} elseif ($audioFiles -ge 40) {"WARN"} else {"FAIL"})

    # Check file sizes (should be > 10MB for 30min+ audio)
    $smallFiles = @(Get-ChildItem "$AUDIO_RAW/*.wav" -ErrorAction SilentlyContinue | Where-Object { $_.Length -lt 10MB }).Count
    Write-QualityCheck "$smallFiles files < 10MB (suspiciously small)" $(if ($smallFiles -eq 0) {"OK"} else {"WARN"})

    $totalSize = (Get-ChildItem "$AUDIO_RAW/*.wav" -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
    Write-Host "`n  Total audio size: $([math]::Round($totalSize / 1GB, 2)) GB"

    if ($StopAfter -eq 0) {
        Write-Host "`n  Stopped after Step 0. Review logs at $LOG_DIR/step0_download.log" -ForegroundColor Yellow
        Write-Host "  Resume with: .\run_pipeline.ps1 -StartFrom 1" -ForegroundColor Yellow
        exit 0
    }
}

# =========================================================================
# STEP 0b: Split Long Audio
# =========================================================================
if ($StartFrom -le 1 -and $StopAfter -ge 1) {
    Write-StepHeader "0b" "Split Long Audio into ~10 min segments (Silero VAD)"

    Run-PythonStep "scripts/00b_split_audio.py --input $AUDIO_RAW --output $AUDIO_DIR --segment-min 10" "$LOG_DIR/step0b_split.log"
    if ($script:_lastPythonExit -ne 0) { Write-Host "STEP 0b FAILED" -ForegroundColor Red; exit 1 }

    # Quality check
    $splitFiles = @(Get-ChildItem "$AUDIO_DIR/*.wav" -ErrorAction SilentlyContinue).Count
    $rawFiles = @(Get-ChildItem "$AUDIO_RAW/*.wav" -ErrorAction SilentlyContinue).Count
    $ratio = if ($rawFiles -gt 0) { [math]::Round($splitFiles / $rawFiles, 1) } else { 0 }
    Write-Host "`n  --- Quality Check: Split ---"
    Write-QualityCheck "$splitFiles segments from $rawFiles files (avg $ratio segments/file)" $(if ($splitFiles -gt $rawFiles) {"OK"} else {"WARN"})

    $totalSize = (Get-ChildItem "$AUDIO_DIR/*.wav" -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
    Write-QualityCheck "Total split audio: $([math]::Round($totalSize / 1GB, 2)) GB" "OK"

    if ($StopAfter -eq 1) {
        Write-Host "`n  Stopped after Step 0b. Review logs at $LOG_DIR/step0b_split.log" -ForegroundColor Yellow
        Write-Host "  Resume with: .\run_pipeline.ps1 -StartFrom 2" -ForegroundColor Yellow
        exit 0
    }
}

# =========================================================================
# STEP 1: Speaker Diarization (GPU, slow)
# =========================================================================
if ($StartFrom -le 2 -and $StopAfter -ge 2) {
    Write-StepHeader 1 "Speaker Diarization (pyannote, GPU)"
    Write-Host "  Processing segments in $AUDIO_DIR ..."
    Write-Host "  (Estimated: ~2-5 min per segment, needs ~2GB VRAM)"

    Run-PythonStep "scripts/01_diarize.py --input $AUDIO_DIR --output $RTTM_DIR --num-speakers 2" "$LOG_DIR/step1_diarize.log"
    if ($script:_lastPythonExit -ne 0) { Write-Host "STEP 1 FAILED" -ForegroundColor Red; exit 1 }

    # Quality check: RTTM files
    $rttmFiles = @(Get-ChildItem "$RTTM_DIR/*.rttm" -ErrorAction SilentlyContinue).Count
    $splitFiles = @(Get-ChildItem "$AUDIO_DIR/*.wav" -ErrorAction SilentlyContinue).Count
    Write-Host "`n  --- Quality Check: Diarization ---"
    Write-QualityCheck "$rttmFiles RTTM files (expected $splitFiles)" $(if ($rttmFiles -eq $splitFiles) {"OK"} elseif ($rttmFiles -gt 0) {"WARN"} else {"FAIL"})

    # Check RTTM content: each file should have exactly 2 speakers
    $badSpeakers = 0
    Get-ChildItem "$RTTM_DIR/*.rttm" -ErrorAction SilentlyContinue | ForEach-Object {
        $speakers = @(Get-Content $_.FullName | ForEach-Object { ($_ -split '\s+')[7] } | Sort-Object -Unique).Count
        if ($speakers -ne 2) { $badSpeakers++ }
    }
    Write-QualityCheck "$badSpeakers files with != 2 speakers" $(if ($badSpeakers -eq 0) {"OK"} elseif ($badSpeakers -le 5) {"WARN"} else {"FAIL"})

    if ($StopAfter -eq 2) {
        Write-Host "`n  Stopped after Step 1. Review: spot-check RTTM files in $RTTM_DIR" -ForegroundColor Yellow
        Write-Host "  Resume with: .\run_pipeline.ps1 -StartFrom 3" -ForegroundColor Yellow
        exit 0
    }
}

# =========================================================================
# STEP 2: Build VA Matrix (CPU, fast)
# =========================================================================
if ($StartFrom -le 3 -and $StopAfter -ge 3) {
    Write-StepHeader 2 "Build Voice Activity Matrix"

    Run-PythonStep "scripts/02_build_va_matrix.py --rttm-dir $RTTM_DIR --audio-dir $AUDIO_DIR --output $VA_DIR" "$LOG_DIR/step2_va.log"
    if ($script:_lastPythonExit -ne 0) { Write-Host "STEP 2 FAILED" -ForegroundColor Red; exit 1 }

    $vaFiles = @(Get-ChildItem "$VA_DIR/*.pt" -ErrorAction SilentlyContinue).Count
    Write-Host "`n  --- Quality Check: VA Matrix ---"
    Write-QualityCheck "$vaFiles VA matrix files created" $(if ($vaFiles -gt 0) {"OK"} else {"FAIL"})

    if ($StopAfter -eq 3) {
        Write-Host "`n  Resume with: .\run_pipeline.ps1 -StartFrom 4" -ForegroundColor Yellow
        exit 0
    }
}

# =========================================================================
# STEP 3: Generate VAP Labels (CPU, fast)
# =========================================================================
if ($StartFrom -le 4 -and $StopAfter -ge 4) {
    Write-StepHeader 3 "Generate VAP Labels (256-class)"

    Run-PythonStep "scripts/03_generate_labels.py --input $VA_DIR --output $LABEL_DIR" "$LOG_DIR/step3_labels.log"
    if ($script:_lastPythonExit -ne 0) { Write-Host "STEP 3 FAILED" -ForegroundColor Red; exit 1 }

    $labelFiles = @(Get-ChildItem "$LABEL_DIR/*.pt" -ErrorAction SilentlyContinue).Count
    Write-Host "`n  --- Quality Check: VAP Labels ---"
    Write-QualityCheck "$labelFiles label files created" $(if ($labelFiles -gt 0) {"OK"} else {"FAIL"})

    if ($StopAfter -eq 4) {
        Write-Host "`n  Resume with: .\run_pipeline.ps1 -StartFrom 5" -ForegroundColor Yellow
        exit 0
    }
}

# =========================================================================
# STEP 4: ASR Transcription (GPU, slow)
# =========================================================================
if ($StartFrom -le 5 -and $StopAfter -ge 5) {
    Write-StepHeader 4 "ASR Transcription (Whisper large-v3, GPU)"
    Write-Host "  Processing segments in $AUDIO_DIR ..."
    Write-Host "  (Estimated: ~5-10 min per segment, needs ~3GB VRAM)"

    Run-PythonStep "scripts/04_transcribe.py --input $AUDIO_DIR --output $TRANSCRIPT_DIR --model large-v3" "$LOG_DIR/step4_transcribe.log"
    if ($script:_lastPythonExit -ne 0) { Write-Host "STEP 4 FAILED" -ForegroundColor Red; exit 1 }

    $transcriptFiles = @(Get-ChildItem "$TRANSCRIPT_DIR/*.json" -ErrorAction SilentlyContinue).Count
    $splitFiles = @(Get-ChildItem "$AUDIO_DIR/*.wav" -ErrorAction SilentlyContinue).Count
    Write-Host "`n  --- Quality Check: Transcription ---"
    Write-QualityCheck "$transcriptFiles transcripts (expected $splitFiles)" $(if ($transcriptFiles -eq $splitFiles) {"OK"} elseif ($transcriptFiles -gt 0) {"WARN"} else {"FAIL"})

    # Check for empty transcripts
    $emptyTranscripts = 0
    Get-ChildItem "$TRANSCRIPT_DIR/*.json" -ErrorAction SilentlyContinue | ForEach-Object {
        $size = $_.Length
        if ($size -lt 100) { $emptyTranscripts++ }
    }
    Write-QualityCheck "$emptyTranscripts empty/tiny transcripts (<100 bytes)" $(if ($emptyTranscripts -eq 0) {"OK"} elseif ($emptyTranscripts -le 3) {"WARN"} else {"FAIL"})

    if ($StopAfter -eq 5) {
        Write-Host "`n  Stopped after Step 4. Spot-check transcripts in $TRANSCRIPT_DIR" -ForegroundColor Yellow
        Write-Host "  Resume with: .\run_pipeline.ps1 -StartFrom 6" -ForegroundColor Yellow
        exit 0
    }
}

# =========================================================================
# STEP 5: Text-Frame Alignment (CPU, fast)
# =========================================================================
if ($StartFrom -le 6 -and $StopAfter -ge 6) {
    Write-StepHeader 5 "Text-Frame Alignment"

    Run-PythonStep "scripts/05_align_text.py --transcripts $TRANSCRIPT_DIR --va-matrices $VA_DIR --output $TEXT_DIR" "$LOG_DIR/step5_align.log"
    if ($script:_lastPythonExit -ne 0) { Write-Host "STEP 5 FAILED" -ForegroundColor Red; exit 1 }

    $textFiles = @(Get-ChildItem "$TEXT_DIR/*.json" -ErrorAction SilentlyContinue).Count
    Write-Host "`n  --- Quality Check: Text Alignment ---"
    Write-QualityCheck "$textFiles text frame files created" $(if ($textFiles -gt 0) {"OK"} else {"FAIL"})

    if ($StopAfter -eq 6) {
        Write-Host "`n  Resume with: .\run_pipeline.ps1 -StartFrom 7" -ForegroundColor Yellow
        exit 0
    }
}

# =========================================================================
# STEP 6: Create Manifest (train/val/test split)
# =========================================================================
if ($StartFrom -le 7 -and $StopAfter -ge 7) {
    Write-StepHeader 6 "Create Manifest (80/10/10 split)"

    Run-PythonStep "scripts/06_create_manifest.py --audio-dir $AUDIO_DIR --va-dir $VA_DIR --label-dir $LABEL_DIR --text-dir $TEXT_DIR --output $OUTPUT_DIR" "$LOG_DIR/step6_manifest.log"
    if ($script:_lastPythonExit -ne 0) { Write-Host "STEP 6 FAILED" -ForegroundColor Red; exit 1 }

    # Quality check: manifest file counts
    if (Test-Path "$OUTPUT_DIR/vap_manifest_train.json") {
        $trainCount = @(Get-Content "$OUTPUT_DIR/vap_manifest_train.json" -Raw | ConvertFrom-Json).Count
        $valCount = @(Get-Content "$OUTPUT_DIR/vap_manifest_val.json" -Raw | ConvertFrom-Json).Count
        $testCount = @(Get-Content "$OUTPUT_DIR/vap_manifest_test.json" -Raw | ConvertFrom-Json).Count
        $total = $trainCount + $valCount + $testCount
        Write-Host "`n  --- Quality Check: Manifest ---"
        Write-QualityCheck "Train: $trainCount / Val: $valCount / Test: $testCount (Total: $total)" "OK"

        if ($total -gt 0) {
            $trainPct = [math]::Round($trainCount / $total * 100, 0)
            $valPct = [math]::Round($valCount / $total * 100, 0)
            $testPct = [math]::Round($testCount / $total * 100, 0)
            Write-QualityCheck "Split ratio: ${trainPct}/${valPct}/${testPct}% (target: 80/10/10)" $(if ([math]::Abs($trainPct - 80) -le 5) {"OK"} else {"WARN"})
        }
    }

    if ($StopAfter -eq 7) {
        Write-Host "`n  Resume with: .\run_pipeline.ps1 -StartFrom 8" -ForegroundColor Yellow
        exit 0
    }
}

# =========================================================================
# STEP 7: Validate Dataset Quality
# =========================================================================
if ($StartFrom -le 8 -and $StopAfter -ge 8) {
    Write-StepHeader 7 "Validate Dataset Quality"

    Run-PythonStep "scripts/07_validate_data.py --manifest $OUTPUT_DIR/vap_manifest_train.json" "$LOG_DIR/step7_validate.log"

    Write-Host "`n  Full validation report saved to data/_validation_report.json"
}

# =========================================================================
# SUMMARY
# =========================================================================
Write-Host "`n$("=" * 60)" -ForegroundColor Green
Write-Host "  PIPELINE COMPLETE" -ForegroundColor Green
Write-Host "$("=" * 60)" -ForegroundColor Green
Write-Host ""
Write-Host "  Logs saved to: $LOG_DIR/"
Write-Host "  Validation report: data/_validation_report.json"
Write-Host ""
Write-Host "  Next steps:"
Write-Host "    1. Review validation report and logs"
Write-Host "    2. Run training: python train.py --config configs/default.yaml"
Write-Host "    3. Run evaluation: python evaluate.py --checkpoint outputs/mm_vap/best_model.pt"
Write-Host ""
