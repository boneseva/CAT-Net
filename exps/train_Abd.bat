@echo off
setlocal enabledelayedexpansion
REM Train a model to segment abdominal MRI (T2 fold of CHAOS challenge)

REM GPU ID Configuration
set GPUID1=0
set CUDA_VISIBLE_DEVICES=%GPUID1%

REM Shared configurations
set DATASET=CHAOST2
set NWORKER=0
set RUNS=1
set ALL_EV="2"
set TEST_LABEL=[1,2,3,4]
set EXCLUDE_LABEL=None
set USE_GT=False

REM Training configurations
set NSTEP=100000
set DECAY=0.98

set MAX_ITER=1000
set SNAPSHOT_INTERVAL=20000
set SEED=2021

set PRETRAIN=C:\Users\bonese\Documents\Courses\LRDL\LRDL-project\CAT-Net\resnet101-63fe2227.pth

echo ========================================================================

REM Loop through evaluation folds (you can add more if needed)
for %%E in (%ALL_EV%) do (
    set "EVAL_FOLD=%%E"
    set "PREFIX=train_%DATASET%_cv!EVAL_FOLD!"
    echo !PREFIX!
    set "LOGDIR=%cd%\exps_on_%DATASET%"

    REM Check if log directory exists, if not, create it
    if not exist "!LOGDIR!" (
        mkdir "!LOGDIR!"
    )

    REM Change directory to the location of the training script
    cd /d C:\Users\bonese\Documents\Courses\LRDL\LRDL-project\CAT-Net

    REM Execute the Python training script
    python train_main.py with ^
        mode="train" ^
        dataset=%DATASET% ^
        num_workers=%NWORKER% ^
        n_steps=%NSTEP% ^
        eval_fold=!EVAL_FOLD! ^
        test_label=%TEST_LABEL% ^
        exclude_label=%EXCLUDE_LABEL% ^
        use_gt=%USE_GT% ^
        max_iters_per_load=%MAX_ITER% ^
        seed=%SEED% ^
        save_snapshot_every=%SNAPSHOT_INTERVAL% ^
        lr_step_gamma=%DECAY% ^
        path.log_dir=!LOGDIR! ^
        pretrain_path=%PRETRAIN% ^
)

pause
