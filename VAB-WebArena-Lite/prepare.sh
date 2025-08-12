#!/bin/bash
# re-validate login information
mkdir -p ./.auth
PYTHONPATH=. python browser_env/auto_login.py