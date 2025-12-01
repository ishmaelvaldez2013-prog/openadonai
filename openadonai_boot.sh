#!/bin/zsh
# OpenAdonAI boot script for login auto-start

cd /Users/ishmael/Developer/OpenAdonAI/Tools/rag_service

# No manual export of .env — Python handles this.

# Start the full pipeline: Ollama + models + Oracle API
openadonai start
