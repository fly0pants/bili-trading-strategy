#!/bin/bash

# Deploy script for GitHub Pages

# Colors for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting deployment to GitHub Pages...${NC}"

# Check if git is installed
if ! [ -x "$(command -v git)" ]; then
  echo -e "${RED}Error: git is not installed.${NC}" >&2
  exit 1
fi

# Copy main README to docs
echo -e "${GREEN}Copying README.md to docs directory...${NC}"
cp README.md docs/

# Create necessary directories if they don't exist
mkdir -p docs/img docs/css docs/js

# Copy images to docs/img directory
echo -e "${GREEN}Copying images to docs/img directory...${NC}"
cp -r static/img/* docs/img/ 2>/dev/null || :

# Copy CSS to docs/css directory
echo -e "${GREEN}Copying CSS to docs/css directory...${NC}"
cp -r static/css/* docs/css/ 2>/dev/null || :

# Copy JS to docs/js directory
echo -e "${GREEN}Copying JS to docs/js directory...${NC}"
cp -r static/js/* docs/js/ 2>/dev/null || :

# Initialize gh-pages branch if it doesn't exist
if ! git rev-parse --verify gh-pages >/dev/null 2>&1; then
  echo -e "${GREEN}Creating gh-pages branch...${NC}"
  git checkout --orphan gh-pages
  git rm -rf .
  git commit --allow-empty -m "Initialize gh-pages branch"
  git checkout master
fi

# Switch to gh-pages branch
echo -e "${GREEN}Switching to gh-pages branch...${NC}"
git checkout gh-pages

# Copy docs directory content to the root of gh-pages
echo -e "${GREEN}Copying documentation to gh-pages branch...${NC}"
cp -R docs/* .

# Add, commit, and push changes
echo -e "${GREEN}Committing changes to gh-pages branch...${NC}"
git add .
git commit -m "Update GitHub Pages content"
git push origin gh-pages

# Switch back to master branch
echo -e "${GREEN}Switching back to master branch...${NC}"
git checkout master

echo -e "${GREEN}Deployment complete! Your site should be available at:${NC}"
echo -e "${YELLOW}https://fly0pants.github.io/bili-trading-strategy/${NC}" 