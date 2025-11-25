# OTB100 Evaluation Script - CSRT Baseline
# Runs C++ tracker on all sequences and logs metrics

$ExePath = "e:\SourceCode\C2P\Project\ECR_Tracking\build\ECR_Tracking_otb.exe"
$DatasetPath = "e:\SourceCode\C2P\Project\ECR_Tracking\toolkit\sequences"
$OutputFile = "e:\SourceCode\C2P\Project\ECR_Tracking\otb100_csrt_results.txt"

# Get all sequences
$SequencesFile = Join-Path $DatasetPath "SEQUENCES"
$Sequences = Get-Content $SequencesFile

# Initialize results file
$Header = @"
===========================================================================
OTB100 EVALUATION - CSRT BASELINE (No Rescue)
===========================================================================
Date: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
Total Sequences: $($Sequences.Count)
===========================================================================
Sequence              Frames     Avg IoU    Success@0.50         FPS
---------------------------------------------------------------------------
"@

Set-Content -Path $OutputFile -Value $Header

# Track counters
$TotalSequences = $Sequences.Count
$Completed = 0
$Failed = 0
$TotalIoU = 0
$TotalSuccess = 0
$TotalFPS = 0
$TotalFrames = 0

Write-Host "Starting OTB100 evaluation with $TotalSequences sequences..." -ForegroundColor Cyan
Write-Host ""

foreach ($SeqName in $Sequences) {
    $Completed++
    Write-Host "[$Completed/$TotalSequences] Processing: $SeqName" -ForegroundColor Yellow
    
    # Run tracker (suppress output, only capture summary)
    $Output = & $ExePath --dataset $DatasetPath --sequence $SeqName --no-display 2>&1 | Out-String
    
    # Parse results from output
    if ($Output -match "(\d+) frames processed") {
        $Frames = [int]$Matches[1]
    } else {
        $Frames = 0
    }
    
    if ($Output -match "IoU:\s+([\d.]+)") {
        $IoU = [double]$Matches[1]
    } else {
        $IoU = 0.0
    }
    
    if ($Output -match "Success:\s+([\d.]+)") {
        $Success = [double]$Matches[1]
    } else {
        $Success = 0.0
    }
    
    if ($Output -match "FPS:\s+([\d.]+)") {
        $FPS = [double]$Matches[1]
    } else {
        $FPS = 0.0
    }
    
    # Check if tracking succeeded
    if ($Frames -eq 0 -or $Output -match "Error|Failed") {
        Write-Host "  ❌ FAILED" -ForegroundColor Red
        $ResultLine = "{0,-20}  {1,6}     {2,7}    {3,12}    {4,8} FAILED" -f $SeqName, $Frames, $IoU.ToString("F3"), $Success.ToString("F3"), $FPS.ToString("F2")
        $Failed++
    } else {
        Write-Host "  ✅ Frames: $Frames, IoU: $($IoU.ToString('F3')), Success: $($Success.ToString('F3')), FPS: $($FPS.ToString('F2'))" -ForegroundColor Green
        $ResultLine = "{0,-20}  {1,6}     {2,7}    {3,12}    {4,8}" -f $SeqName, $Frames, $IoU.ToString("F3"), $Success.ToString("F3"), $FPS.ToString("F2")
        
        # Accumulate stats
        $TotalFrames += $Frames
        $TotalIoU += $IoU
        $TotalSuccess += $Success
        $TotalFPS += $FPS
    }
    
    # Append to file
    Add-Content -Path $OutputFile -Value $ResultLine
}

# Calculate averages
$SuccessCount = $TotalSequences - $Failed
if ($SuccessCount -gt 0) {
    $AvgIoU = $TotalIoU / $SuccessCount
    $AvgSuccess = $TotalSuccess / $SuccessCount
    $AvgFPS = $TotalFPS / $SuccessCount
} else {
    $AvgIoU = 0
    $AvgSuccess = 0
    $AvgFPS = 0
}

# Write summary
$Summary = @"

===========================================================================
SUMMARY
===========================================================================
Total Sequences:      $TotalSequences
Completed:            $SuccessCount
Failed:               $Failed
Total Frames:         $TotalFrames
---------------------------------------------------------------------------
Average IoU:          $($AvgIoU.ToString('F3'))
Average Success:      $($AvgSuccess.ToString('F3'))
Average FPS:          $($AvgFPS.ToString('F2'))
===========================================================================
"@

Add-Content -Path $OutputFile -Value $Summary

Write-Host ""
Write-Host "===========================================================================
EVALUATION COMPLETE
===========================================================================
" -ForegroundColor Cyan
Write-Host "Completed:            $SuccessCount / $TotalSequences" -ForegroundColor Green
Write-Host "Failed:               $Failed" -ForegroundColor $(if ($Failed -gt 0) { "Red" } else { "Green" })
Write-Host "Total Frames:         $TotalFrames" -ForegroundColor White
Write-Host "---------------------------------------------------------------------------"
Write-Host "Average IoU:          $($AvgIoU.ToString('F3'))" -ForegroundColor Cyan
Write-Host "Average Success:      $($AvgSuccess.ToString('F3'))" -ForegroundColor Cyan
Write-Host "Average FPS:          $($AvgFPS.ToString('F2'))" -ForegroundColor Cyan
Write-Host "===========================================================================
" -ForegroundColor Cyan
Write-Host "Results saved to: $OutputFile" -ForegroundColor Yellow
