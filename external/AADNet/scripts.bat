@echo off

SET DATASET=aad_data/datasets/DTU& REM path to DTU dataset root

python cross_validate_loso.py -c config/config_AADNet_SI_DTU.yml -j LOSO_AADNet_DTU
python cross_validate_ss.py -c config/config_AADNet_SS_DTU.yml -j SS_AADNet_DTU

REM Pilot (1 subject, 2 folds, 2 epochs):
REM python cross_validate_loso.py -c config/config_AADNet_SI_DTU_pilot.yml -j LOSO_AADNet_DTU_pilot

pause
