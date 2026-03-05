param (
    [string]$Csv,
    [string]$Json
)

if (-not (Test-Path $Csv)) {
    Write-Error "CSV file not found: $Csv"
    exit 1
}

# CSV 로드 (인코딩 무시 및 UTF8 강제)
# NCU 테이블 방식은 헤더가 명확하므로 Import-Csv가 잘 작동합니다.
$csvData = Import-Csv -Path $Csv

# 데이터가 있는 3번째 줄(Index 1, 유닛행 제외)부터 실제 데이터입니다.
# 하지만 Import-Csv는 2행(유닛행)을 데이터로 인식할 수 있으므로 필터링이 필요합니다.
$actualData = $csvData | Where-Object { $_.ID -ne "" }

if ($null -eq $actualData) {
    Write-Warning "No valid data rows found."
    exit 0
}

$row = $actualData[0]
$results = @{
    timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss.fff"
    kernel    = $row."Kernel Name"
    device    = $row."Device"
    grid      = $row."Grid Size"
    block     = $row."Block Size"
    metrics   = @{}
}

# 메트릭으로 추출할 헤더 목록 (수동 지정 또는 자동 추출)
# 공유해주신 헤더 중 수치 데이터인 것들만 골라냅니다.
$metricKeys = @(
    "gpu__time_duration.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active"
)

foreach ($key in $metricKeys) {
    if ($row.$key) {
        $valStr = $row.$key
        $cleanVal = $valStr -replace '[^0-9.]', ''
        
        # 유닛 정보 매핑 (CSV 2행에 있던 유닛 정보 - 보통 고정되어 있음)
        $unit = ""
        if ($key -like "*duration*") { $unit = "us" }
        if ($key -like "*pct*") { $unit = "%" }

        $results.metrics[$key] = @{
            val  = [double]$cleanVal
            unit = $unit
        }
    }
}

# JSON 저장
$results | ConvertTo-Json -Depth 10 | Out-File -FilePath $Json -Encoding utf8
Write-Host "[ps1] Metrics successfully extracted to $Json"