@echo off

SET DATASET=aad_data/datasets/DTU& REM path to DTU dataset root

python cross_validate_loso.py -c config/config_AADNet_SI_DTU.yml -j LOSO_AADNet_DTU
python cross_validate_ss.py -c config/config_AADNet_SS_DTU.yml -j SS_AADNet_DTU

REM Pilot (1 subject, 2 folds, 2 epochs):
REM python cross_validate_loso.py -c config/config_AADNet_SI_DTU_pilot.yml -j LOSO_AADNet_DTU_pilot

REM Optional: Upload AADNet artifacts to GCS (requires gsutil and auth)
SET ARTIFACTS_BUCKET_URI=gs://aad_data/artifacts/aad_xai
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set RUN_ID=%%i
where gsutil >nul 2>nul
if %ERRORLEVEL% EQU 0 (
	gsutil -m rsync -r output %ARTIFACTS_BUCKET_URI%/%RUN_ID%/aadnet_output
	gsutil -m rsync -r results %ARTIFACTS_BUCKET_URI%/%RUN_ID%/aadnet_results
) else (
	echo gsutil not found. Skipping artifact upload.
)

pause
