#!/bin/bash

conda activate /pscratch/sd/z/zlu39/.conda_envs/parquet_server/
PARQUET_VIEWER_FILE=generations/qasper_validation_flat_first100_seed_42.parquet uvicorn parquet_viewer:app --port 19484 --reload