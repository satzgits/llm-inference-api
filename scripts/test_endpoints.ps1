# Test all LLM Inference API endpoints
$BASE = "http://localhost:8008"

Write-Host "=== LLM Inference API Test Suite ===" -ForegroundColor Cyan
Write-Host ""

# 1. Health check
Write-Host "1. GET /health" -ForegroundColor Yellow
try {
    $health = Invoke-RestMethod -Uri "$BASE/health" -ErrorAction Stop
    Write-Host "   Status: $($health.status)" -ForegroundColor Green
    Write-Host "   Ollama: $($health.ollama_connected)" 
} catch {
    Write-Host "   FAILED: $_" -ForegroundColor Red
}

# 2. Generate
Write-Host "`n2. POST /generate" -ForegroundColor Yellow
$body = @{ prompt = "What is quantum computing?"; max_tokens = 100; temperature = 0.1 } | ConvertTo-Json
try {
    $gen = Invoke-RestMethod -Uri "$BASE/generate" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 120
    Write-Host "   Response: $($gen.response)" -ForegroundColor Green
    Write-Host "   Tokens: $($gen.tokens_used)" 
    Write-Host "   Latency: $($gen.inference_time_ms)ms"
} catch {
    Write-Host "   FAILED: $_" -ForegroundColor Red
}

# 3. Embed
Write-Host "`n3. POST /embed" -ForegroundColor Yellow
$body2 = @{ input = "Sample text to embed" } | ConvertTo-Json
try {
    $emb = Invoke-RestMethod -Uri "$BASE/embed" -Method Post -Body $body2 -ContentType "application/json" -TimeoutSec 30
    Write-Host "   Embedding vector length: $($emb.embedding.Count)" -ForegroundColor Green
    Write-Host "   First 3 values: $($emb.embedding[0..2])"
} catch {
    Write-Host "   FAILED: $_" -ForegroundColor Red
}

Write-Host "`n=== All tests complete ===" -ForegroundColor Cyan
