#!/bin/bash
# Script to prepare for Git commits

echo "Preparing for Git commits..."

# Check if git is initialized
if [ ! -d .git ]; then
    echo "Initializing git repository..."
    git init
fi

# Check git config
echo ""
echo "Current Git configuration:"
git config user.name || echo "  user.name: NOT SET"
git config user.email || echo "  user.email: NOT SET"

echo ""
echo "To configure Git (if needed):"
echo "  git config user.name 'Your Name'"
echo "  git config user.email 'your.email@example.com'"

echo ""
echo "Ready for commits!"
echo ""
echo "Next steps:"
echo "1. Configure git user (if needed)"
echo "2. Review GIT_COMMIT_PLAN.md for commit strategy"
echo "3. Start making commits following the plan"

