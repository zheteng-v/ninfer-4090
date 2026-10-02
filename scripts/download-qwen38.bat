@echo off
setlocal
set "ROOT=%~dp0"
set "MODEL_DIR=%ROOT%models"
set "MODEL=%MODEL_DIR%\qwen3_8_27b.ninfer"
rem Pinned to the artifact release commit of the current v3 container (HTTP HEAD verified:
rem x-linked-etag matches EXPECTED_SHA256, content-length 20,437,521,664 bytes).
set "REVISION=1cbd84e7221e51186bd7f093a149912d2489625b"
set "EXPECTED_SHA256=81f924d440c27261d820c19a9f8d45794c5aee410f8a68bd358133fa8c0375da"

if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%"
echo Downloading Qwen3.8-27B NInfer model (revision %REVISION%)...
curl.exe -L -C - --fail --output "%MODEL%" "https://huggingface.co/neroued/Qwen3.8-27B-NInfer/resolve/%REVISION%/qwen3_8_27b.ninfer"
if errorlevel 1 (
  echo Download failed. Run this file again to resume.
  exit /b 1
)
echo Expected SHA-256: %EXPECTED_SHA256%
echo Verify with: certutil -hashfile "%MODEL%" SHA256
echo Model ready: %MODEL%
