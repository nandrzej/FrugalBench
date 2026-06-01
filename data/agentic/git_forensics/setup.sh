#!/bin/bash
# Setup a git repo with a leaked password in commit history
cd /workspace/git_forensics
git init --initial-branch=main
git config user.email "dev@example.com"
git config user.name "Developer"

echo "# Project Config" > README.md
git add README.md && git commit -m "Initial commit"

echo "DB_HOST=localhost" > config.env
echo "DB_PASSWORD=SuperSecret42" >> config.env
git add config.env && git commit -m "Add database config"

echo "DB_HOST=localhost" > config.env
echo "DB_PASSWORD=changeme" >> config.env
git add config.env && git commit -m "Remove sensitive credentials"

echo "print('hello world')" > app.py
git add app.py && git commit -m "Add application code"
